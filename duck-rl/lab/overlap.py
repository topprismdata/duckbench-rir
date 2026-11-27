"""The PPO update of lab.learn.train: on the CPU (rsl_rl) or on the Apple GPU (MLX, own process), serial or
overlapped with the next rollout (kit v2, T1; measured in docs/research/novel-moves/rl-code/mac-speed/t1t2/).

Updater is what learn.train calls once per collected batch:
  learner "cpu": rsl_rl PPO.update on the CPU copy `alg`, serial: the pre-v2 lab bit for bit (deterministic given the
                 seed);
  learner "mlx": MlxLearner, the same update re-implemented in MLX (lab/mlx_ppo.py) in a server process (no GIL
                 sharing with the rollout, no torch). Overlapped (the default with mlx), with decoupled PPO whose
                 importance weights are capped at `iw_cap`.
Any MLX failure (not importable, the process dies or goes silent, weights that do not load) prints one warning line
and the run continues on the CPU learner from the last synced weights; the batch in flight is lost.

The rollout (act -> env.step -> process_env_step -> compute_returns) always runs on the CPU copy `alg`.
Overlapped loop order per iteration k: collect B_k with the CPU weights -> compute_returns (behaviour critic) ->
wait() for the update of B_{k-1} and load its weights -> submit(B_k). So B_k is collected with the weights from before
the previous update and trained from the weights after it: one iteration of policy lag. The plain lag slows learning
per step (the adaptive-KL LR sees the previous update's shift and pins at the floor; ~1/4 of the samples start past
the clip), so the overlapped MLX update is decoupled PPO (mlx_ppo.Learner._decouple): clip and LR around the policy at
the start of the update, stale samples reweighted by w = pi_start / pi_behaviour, w capped at iw_cap.

GpuLearner (research only, t1t2/run.py): a second rsl_rl PPO on MPS, serial or in a thread. No gain (rsl_rl's per-op
dispatch; 253 ms per update, M).
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .batch_layout import Layout, layout_of_params, pack_columns

PC = time.perf_counter
_FIELDS = ("actions", "rewards", "dones", "values", "actions_log_prob", "returns", "advantages")


def _cpu_state(module: torch.nn.Module) -> dict:
    return {k: v.detach().to("cpu", copy=True) for k, v in module.state_dict().items()}


class GpuLearner:
    def __init__(self, alg_cpu, env, move, init_std: float, device: str = "mps", overlap: bool = False):
        from .learn import build
        self.cpu, self.device, self.overlap = alg_cpu, device, overlap
        self.alg = build(env, move, device, init_std)  # own actor/critic/Adam/storage on the GPU
        self.alg.actor.load_state_dict(alg_cpu.actor.state_dict())
        self.alg.critic.load_state_dict(alg_cpu.critic.state_dict())
        self.alg.train_mode()
        self.exec = ThreadPoolExecutor(1, thread_name_prefix="gpu-update") if overlap else None
        self.pending = None
        self.t_handoff = 0.0

    # -- config (only while the GPU copy is idle, i.e. right before submit) ---------------------------------------
    def set_actor_trainable(self, on: bool) -> None:
        for p in self.alg.actor.parameters():
            p.requires_grad_(on)

    def set_lr(self, schedule: str, lr: float) -> None:
        self.alg.schedule, self.alg.learning_rate = schedule, lr
        for g in self.alg.optimizer.param_groups:
            g["lr"] = lr

    @property
    def learning_rate(self) -> float:
        return self.alg.learning_rate

    @property
    def busy(self) -> bool:
        return self.pending is not None

    # -- the update ---------------------------------------------------------------------------------------------
    def submit(self) -> None:
        assert self.pending is None, "wait() for the previous update first"
        t = PC()
        src, dst = self.cpu.storage, self.alg.storage
        for name in _FIELDS:
            getattr(dst, name).copy_(getattr(src, name))
        for k in src.observations.keys():
            dst.observations[k].copy_(src.observations[k])
        if dst.distribution_params is None:
            dst.distribution_params = tuple(torch.empty_like(p, device=self.device) for p in src.distribution_params)
        for d, s in zip(dst.distribution_params, src.distribution_params):
            d.copy_(s)
        dst.step = src.step
        if self.device == "mps":
            torch.mps.synchronize()  # the copies are done before the CPU storage is overwritten by the next rollout
        src.clear()  # (rsl_rl's update clears it; the CPU copy never updates)
        self.t_handoff = PC() - t
        self.pending = self.exec.submit(self._update) if self.overlap else self._update()

    def _update(self):
        t = PC()
        losses = self.alg.update()
        state = (_cpu_state(self.alg.actor), _cpu_state(self.alg.critic))  # .to("cpu") waits for the GPU
        return losses, state, PC() - t

    def wait(self):
        """-> (losses, t_update_on_gpu, t_blocked). Loads the updated weights into the CPU copy."""
        t = PC()
        res = self.pending.result() if self.overlap else self.pending
        self.pending = None
        losses, (a, c), t_gpu = res
        t_blocked = PC() - t
        self.cpu.actor.load_state_dict(a)
        self.cpu.critic.load_state_dict(c)
        return losses, t_gpu, t_blocked

    def close(self) -> None:
        if self.exec is not None:
            self.exec.shutdown(wait=True)


# ---- the same interface with the update in MLX (lab/mlx_ppo.py) in a separate process -----------------------------------
MLX_PY = sys.executable  # the lab venv (requirements.txt installs mlx on darwin-arm64); the server imports no torch


def pack_params(alg) -> dict:
    """CPU rsl_rl actor/critic -> the numpy arrays mlx_ppo.Learner starts from (names mirror the state_dicts)."""
    a, c = alg.actor.state_dict(), alg.critic.state_dict()
    p = {f"a.{k}": v.numpy().copy() for k, v in a.items() if k.startswith("mlp.")}
    p.update({f"c.{k}": v.numpy().copy() for k, v in c.items() if k.startswith("mlp.")})
    p["a.log_std"] = a["distribution.log_std_param"].numpy().copy()
    nz = alg.actor.obs_normalizer
    p["a.sub"], p["a.div"] = a["obs_normalizer._mean"].numpy().copy(), (a["obs_normalizer._std"] + nz.eps).numpy().copy()
    p["c.mean"], p["c.var"], p["c.std"] = (c[f"obs_normalizer._{k}"].numpy().copy() for k in ("mean", "var", "std"))
    p["c.count"] = int(c["obs_normalizer.count"])
    return p


def pack_batch(st, layout: Layout | None = None) -> np.ndarray:
    """rsl_rl RolloutStorage (after compute_returns) -> float32 [T*N, layout.width] in mlx_ppo's column layout
    (lab/batch_layout.py: any obs width; the stock 61-input move's is 113 columns)."""
    obs = st.observations
    cols = [obs["policy"], obs["critic"], st.actions, st.distribution_params[0], st.distribution_params[1],
            st.actions_log_prob, st.advantages, st.values, st.returns]
    n = st.num_transitions_per_env * st.num_envs
    return pack_columns([t.numpy() for t in cols], n, layout)


class LearnerFailed(RuntimeError):
    """The MLX learner is not importable, its process died or went silent, or its weights do not load."""


def mlx_available() -> bool:
    """MLX is installed (requirements.txt: darwin-arm64 only) and this is Apple silicon."""
    import importlib.util
    import platform
    return sys.platform == "darwin" and platform.machine() == "arm64" and importlib.util.find_spec("mlx") is not None


class MlxLearner:
    """GpuLearner's interface; the update runs in an MLX server process (its own interpreter: no GIL sharing with the
    rollout, no torch). Overlapped: submit() only sends the batch, wait() receives the new weights. Transport or
    sync problems raise LearnerFailed."""

    def __init__(self, alg_cpu, overlap: bool = False, seed: int = 1, python: str = MLX_PY, decoupled: bool = False,
                 iw_cap: float | None = None, timeout: float = 60.0, knobs: dict | None = None,
                 opt_state: dict | None = None):
        """`knobs`: the per-run update knobs sent with every update (mlx_ppo's dkl, value_clip, critic_epochs,
        std_floor; absent = its constants); `opt_state`: a saved Adam state (mlx_ppo opt_state()) to start from."""
        import socket
        import subprocess
        from multiprocessing.connection import Connection
        from pathlib import Path
        self.cpu, self.overlap, self.timeout = alg_cpu, overlap, timeout
        try:  # the batch's columns follow the nets' widths (any obs width: an ENCODER move's 61 + its numbers)
            self.layout = layout_of_params(pack_params(alg_cpu))
        except ValueError as e:
            raise LearnerFailed(f"MLX learner: {e}") from None
        s1, s2 = socket.socketpair()
        for s in (s1, s2):  # the 3.7 MB batch in a few syscalls
            s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self.proc = subprocess.Popen([python, str(Path(__file__).with_name("mlx_ppo.py")), str(s2.fileno())],
                                     pass_fds=[s2.fileno()])
        s2.close()
        self.conn = Connection(s1.detach())
        try:
            self._send(("init", dict(pack_params(alg_cpu), **(opt_state or {})), seed))
            reply = self._recv()[0]
            if reply != "ok" and opt_state:  # a saved Adam state that does not fit: start the MLX learner fresh
                print(f"[resume] the saved MLX Adam state was not restored ({reply}): Adam starts fresh", flush=True)
                self._send(("init", pack_params(alg_cpu), seed))
                reply = self._recv()[0]
            if reply != "ok":
                raise LearnerFailed(f"MLX process did not initialise ({reply})")
        except LearnerFailed:
            self.proc.kill()
            raise
        self.train_actor, self.adaptive, self.lr = True, alg_cpu.schedule == "adaptive", alg_cpu.learning_rate
        self._lr_set = True
        self.critic_lr = None  # the critic's own LR (set_lr; None: the actor's)
        self.knobs = dict(knobs or {})  # (only non-default knobs: an absent one is mlx_ppo's constant)
        self.pending, self.t_handoff, self.decoupled, self.iw_cap = False, 0.0, decoupled, iw_cap
        self._early = None  # an in-flight update's result received early (opt_state), for wait()

    def set_actor_trainable(self, on: bool) -> None:
        self.train_actor = on

    def set_lr(self, schedule: str, lr: float, critic_lr: float | None = None) -> None:
        """Takes effect at the next submit (also when called while an update is in flight). `critic_lr`: the critic's
        own fixed LR (None: the actor's, as before)."""
        self.adaptive, self.lr, self._lr_set, self.critic_lr = schedule == "adaptive", lr, True, critic_lr

    def load_params(self) -> None:
        """The CPU copy's weights and critic normaliser -> the MLX learner (the optimiser state and its RNG stay).
        Only while no update is in flight."""
        assert not self.pending
        self._send(("load", pack_params(self.cpu), None))
        if self._recv()[0] != "ok":
            raise LearnerFailed("MLX process did not load the weights")

    def opt_state(self) -> dict:
        """Both Adam states of the MLX learner as numpy (mlx_ppo opt_state()). While an update is in flight this
        first receives its result (held for wait(): the CPU copy is not touched), and the state is the one after it,
        one update ahead of the CPU copy's weights."""
        if self.pending and self.overlap and self._early is None:
            self._early = self._recv()
        self._send(("opt_state", None, None))
        return self._recv()[0]

    @property
    def learning_rate(self) -> float:
        return self.lr

    @property
    def busy(self) -> bool:
        return self.pending

    def _send(self, msg) -> None:
        try:
            self.conn.send(msg)
        except OSError as e:
            raise LearnerFailed(f"MLX process gone ({e.__class__.__name__}, exit code {self.proc.poll()})") from None

    def _recv(self):
        t = PC()
        try:
            if not self.conn.poll(self.timeout):
                raise LearnerFailed(f"MLX process silent for {self.timeout:.0f} s")
            res = self.conn.recv()
        except (OSError, EOFError) as e:
            raise LearnerFailed(f"MLX process gone ({e.__class__.__name__}, exit code {self.proc.poll()})") from None
        return res, PC() - t

    def submit(self) -> None:
        assert not self.pending
        t = PC()
        try:
            batch = pack_batch(self.cpu.storage, self.layout)
        except ValueError as e:
            raise LearnerFailed(f"MLX learner: {e}") from None
        cfg = dict(train_actor=self.train_actor, adaptive=self.adaptive, lr=self.lr if self._lr_set else None,
                   perm=getattr(self, "_perm", None),  # (_perm: fixed minibatch order, equivalence test only)
                   decoupled=self.decoupled, iw_cap=self.iw_cap,  # decoupled PPO for the lag (mlx_ppo.Learner._decouple)
                   **self.knobs, **({"critic_lr": self.critic_lr} if self.critic_lr is not None else {}))
        self._send(("update", batch, cfg))
        self._lr_set = False
        self.cpu.storage.clear()  # (only once the batch is handed off)
        self.t_handoff = PC() - t
        if not self.overlap:
            self._res = self._recv()
        self.pending = True

    def wait(self):
        """-> (losses, t_update_on_gpu, t_blocked). Loads the updated weights into the CPU copy (all or nothing)."""
        if self._early is not None:
            (losses, lr, p, t_gpu), t_blocked = self._early
            self._early = None
        else:
            (losses, lr, p, t_gpu), t_blocked = self._recv() if self.overlap else self._res
        self.pending = False
        a, c = self.cpu.actor.state_dict(), self.cpu.critic.state_dict()
        new_a = {k: torch.from_numpy(p[f"a.{k}"]) for k in a if k.startswith("mlp.")}
        new_a["distribution.log_std_param"] = torch.from_numpy(p["a.log_std"])
        new_c = {k: torch.from_numpy(p[f"c.{k}"]) for k in c if k.startswith("mlp.")}
        new_c.update({f"obs_normalizer._{k}": torch.from_numpy(p[f"c.{k}"]) for k in ("mean", "var", "std")})
        for old, new in ((a, new_a), (c, new_c)):
            for k, v in new.items():
                if v.shape != old[k].shape or not bool(torch.isfinite(v).all()):
                    raise LearnerFailed(f"MLX weights do not load ({k}: shape {tuple(v.shape)}, finite "
                                        f"{bool(torch.isfinite(v).all())})")
        a.update(new_a)
        c.update(new_c)
        c["obs_normalizer.count"] = torch.tensor(int(p["c.count"]), dtype=torch.long)
        self.cpu.actor.load_state_dict(a)
        self.cpu.critic.load_state_dict(c)
        if not self._lr_set:  # (a set_lr() during the update wins)
            self.lr = lr
        return losses, t_gpu, t_blocked

    def close(self) -> None:
        import subprocess
        try:
            self.conn.send(("close", None, None))
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _cpu_set(alg, train_actor: bool, schedule: str, lr: float, critic_lr: float | None = None) -> None:
    """The pre-v2 release of the actor after the critic warm-up (requires_grad, schedule, LR in every Adam group);
    `critic_lr` -> the critic's group of a lab.learn.TwoLrAdam (CRITIC_LR)."""
    for p in alg.actor.parameters():
        p.requires_grad_(train_actor)
    alg.schedule, alg.learning_rate = schedule, lr
    for g in alg.optimizer.param_groups:
        g["lr"] = lr
    if critic_lr is not None or hasattr(alg.optimizer, "critic_lr"):
        assert hasattr(alg.optimizer, "critic_lr"), "critic_lr needs lab.learn.TwoLrAdam"
        alg.optimizer.critic_lr = critic_lr


class Updater:
    """The update step of lab.learn.train (see the module docstring). update() once per collected batch, after
    compute_returns; finish() before export (the final sync); close() always."""

    # knobs only the MLX update has (lab.learn.train refuses them with learner "cpu"; a fallback drops them, loudly)
    MLX_ONLY = ("critic_epochs", "value_clip")

    def __init__(self, alg, learner: str = "cpu", overlap: bool = False, seed: int = 1, iw_cap: float | None = 3.0,
                 warn=lambda msg: print(msg, flush=True), knobs: dict | None = None, opt_state: dict | None = None):
        """`knobs`: the non-default update knobs (mlx_ppo: dkl, value_clip, critic_epochs, std_floor; std_floor also
        on the CPU learner, the rest of the CPU learner's comes from its rsl_rl cfg); `opt_state`: a saved MLX Adam
        state to start from (--resume)."""
        assert learner in ("cpu", "mlx") and (learner == "mlx" or not overlap), (learner, overlap)
        self.alg, self.warn, self.mlx, self.fallback = alg, warn, None, None
        self.train_actor, self.schedule, self.lr, self.critic_lr = False, alg.schedule, alg.learning_rate, None
        self.knobs = dict(knobs or {})
        self.opt_snapshot = None  # the MLX Adam state fetched between wait() and submit() (update(fetch_opt=True))
        self.threads = torch.get_num_threads()
        self._clamp_std()
        if learner == "mlx":
            try:
                if not mlx_available():
                    raise LearnerFailed("mlx is not importable here (Apple silicon + requirements.txt)")
                self.mlx = MlxLearner(alg, overlap=overlap, seed=seed, decoupled=overlap,
                                      iw_cap=iw_cap if overlap else None, knobs=self.knobs, opt_state=opt_state)
                self.mlx.set_actor_trainable(False)
                torch.set_num_threads(1)  # this process only runs the rollout's inference (as measured in t1t2/)
            except LearnerFailed as e:
                self._fall_back(e)
        for p in alg.actor.parameters():  # critic warm-up first (the CPU copy mirrors it for a fallback)
            p.requires_grad_(False)

    def _clamp_std(self) -> None:
        """STD_FLOOR on the CPU copy (the MLX learner clamps its own after each update)."""
        floor = self.knobs.get("std_floor")
        if floor:
            import math
            with torch.no_grad():
                self.alg.actor.distribution.log_std_param.clamp_(min=math.log(float(floor)))

    @property
    def name(self) -> str:
        return "mlx" if self.mlx else "cpu"

    @property
    def overlap(self) -> bool:
        return bool(self.mlx and self.mlx.overlap)

    @property
    def busy(self) -> bool:
        """An update is in flight: the CPU copy's weights do not include the last submitted batch yet."""
        return bool(self.mlx and self.mlx.busy)

    @property
    def learning_rate(self) -> float:
        return self.mlx.learning_rate if self.mlx else self.alg.learning_rate

    def release(self, schedule: str, lr: float, critic_lr: float | None = None) -> None:
        """End of the critic warm-up: the actor trains from the next submitted batch on. `critic_lr`: the critic's own
        fixed LR from now on (None: the actor's, as before)."""
        self.train_actor = True
        if self.mlx:
            self.mlx.set_actor_trainable(True)
        self.set_lr(schedule, lr, critic_lr)

    def set_lr(self, schedule: str, lr: float, critic_lr: float | None = None) -> None:
        """The LR schedule from the next submitted batch on (release; LR_ANNEAL calls it every iteration)."""
        self.schedule, self.critic_lr = schedule, critic_lr
        if self.mlx:
            self.mlx.set_lr(schedule, lr, critic_lr)
        else:
            _cpu_set(self.alg, self.train_actor, schedule, lr, critic_lr)

    def load_params(self) -> None:
        """The CPU copy's weights -> the learner (MLX; the CPU learner trains the CPU copy itself)."""
        if self.mlx:
            try:
                self.mlx.load_params()
            except LearnerFailed as e:
                self._fall_back(e)

    def opt_state(self) -> dict | None:
        """The MLX learner's Adam states for a checkpoint (None on the CPU learner: its optimizer state dict is in the
        checkpoint already): the snapshot update(fetch_opt=True) took, which matches the CPU copy's weights, else
        fetched now (while an update is in flight: the state after it, one update ahead of the weights)."""
        snap, self.opt_snapshot = self.opt_snapshot, None
        if snap is not None or not self.mlx:
            return snap
        try:
            return self.mlx.opt_state()
        except LearnerFailed as e:
            self._fall_back(e)
            return None

    def update(self, fetch_opt: bool = False):
        """-> (losses | None, t_update, t_blocked). Serial: this batch's update. Overlapped: wait for the previous
        batch's update (its weights -> the CPU copy), start this one, return the previous update's losses.
        `fetch_opt`: also take the MLX Adam state between the wait and the submit (= the CPU copy's weights' state;
        a checkpoint follows: opt_state())."""
        if self.mlx:
            try:
                last = self.mlx.wait() if self.mlx.busy else None
                self.opt_snapshot = self.mlx.opt_state() if fetch_opt and self.mlx.overlap else None
                self.mlx.submit()
                if not self.mlx.overlap:
                    last = self.mlx.wait()
                    self.opt_snapshot = self.mlx.opt_state() if fetch_opt else None
                return last if last else (None, 0.0, 0.0)
            except LearnerFailed as e:
                self._fall_back(e)
                if self.alg.storage.step == 0:  # this batch went to the dead process
                    return None, 0.0, 0.0
        t = PC()
        losses = self.alg.update()
        self._clamp_std()
        dt = PC() - t
        return losses, dt, dt

    def finish(self):
        """Overlapped: wait for the last update, so the CPU copy holds the final weights (-> its losses or None)."""
        self.opt_snapshot = None  # (a snapshot of the weights before the last update: stale now)
        if self.mlx and self.mlx.busy:
            try:
                return self.mlx.wait()
            except LearnerFailed as e:
                self._fall_back(e)
        return None

    def close(self) -> None:
        if self.mlx:
            self.mlx.close()
        torch.set_num_threads(self.threads)

    def _fall_back(self, e: Exception) -> None:
        lr = self.mlx.learning_rate if self.mlx else self.lr
        if self.mlx:
            self.mlx.close()
            self.mlx = None
        torch.set_num_threads(self.threads)
        _cpu_set(self.alg, self.train_actor, self.schedule, lr, self.critic_lr)
        self.fallback = str(e)
        lost = [k for k in self.MLX_ONLY if k in self.knobs]
        self.warn(f"WARNING: MLX learner failed: {e}. Continuing on the CPU learner (rsl_rl, serial) from the last "
                  "synced weights." + (f" MLX-only knobs no longer apply: {', '.join(lost)}." if lost else ""))
