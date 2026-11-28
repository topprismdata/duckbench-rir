"""rsl_rl (rsl-rl-lib 5.5.1) PPO over the CPU env pool, warm-started from a stock skill, + ONNX export.

The adapter is PoolVecEnv (rsl_rl VecEnv over the env pool, reward from the move's reward_terms in the main
process). The pool is lab.fastenv.FastPool (`env="fast"`, the default: numba obs + tracker state, bit-identical to
the reference, checked by lab/selftest.py) or lab.pool.Pool (`env="ref"`, lab/duck.py DuckEnv). The loop is
OnPolicyRunner.learn's inner loop (act -> env.step -> process_env_step -> compute_returns -> update) with a wall-clock
budget, critic-only warm-up, the move's weight curriculum and our own per-iteration log. The update is
lab.overlap.Updater: rsl_rl on the CPU (serial) or MLX on the GPU (overlapped with the next collect).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from . import PLANT
from .duck import DT, unpack
from .fastenv import FastPool
from .overlap import Updater
from .pool import Pool
from .probestop import ProbeStop
from .spec import Move

HIDDEN = [512, 256, 128]
if os.environ.get("DUCK_HIDDEN"):  # H-S3 capacity arm (kit patch, disclosed in kit_diff.patch)
    HIDDEN[:] = [int(x) for x in os.environ["DUCK_HIDDEN"].split(",")]


class PoolVecEnv(VecEnv):
    def __init__(self, move: Move, workers: int, per_worker: int, seed: int, env: str = "fast"):
        assert env in ("fast", "ref"), env
        self.move, self.pool = move, (FastPool if env == "fast" else Pool)(move.env_cfg(), workers, per_worker, seed)
        self.num_envs, self.num_actions = self.pool.n, 14
        self.max_episode_length = int(round(move.episode_s / DT))
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.device, self.cfg = "cpu", move.env_cfg()
        self.weights = move.weights(0.0)
        self._obs = self._td(*self.pool.reset())
        self.acc = None

    @staticmethod
    def _td(obs, priv):
        return TensorDict({"policy": torch.from_numpy(obs), "critic": torch.from_numpy(priv)}, batch_size=[obs.shape[0]])

    def get_observations(self) -> TensorDict:
        return self._obs

    def step(self, actions: torch.Tensor):
        obs, priv, S, term, tout, eps, k = self.pool.step(actions.detach().cpu().numpy().astype(np.float32))
        duck = unpack(S)
        terms = self.move.reward_terms(duck)
        n = S.shape[0]
        r = np.zeros(n)
        for name, v in terms.items():
            v = np.broadcast_to(np.asarray(v, np.float64), (n,))
            r += self.weights[name] * v
            a = self.acc["terms"]
            a[name] = a.get(name, 0.0) + float(v.sum())
            self.acc["tmax"][name] = max(self.acc["tmax"].get(name, -np.inf), float(v.max()))
            self.acc["tmin"][name] = min(self.acc["tmin"].get(name, np.inf), float(v.min()))
        self.acc["n"] += n
        self.acc["vx"] += float(duck["vx"].sum())
        self.acc["yaw_rate"] += float(np.abs(duck["yaw_rate"]).sum())
        self.acc["falls"] += int(duck["fell"].sum())
        self.acc["terms_other"] += int((term & (duck["fell"] == 0)).sum())
        self.acc["eps"] += eps
        self.episode_length_buf = torch.from_numpy(k.astype(np.int64))
        self._obs = self._td(obs, priv)
        dones = torch.from_numpy(term | tout)
        return self._obs, torch.from_numpy((r * DT).astype(np.float32)), dones, {
            "time_outs": torch.from_numpy(tout.astype(np.float32))}

    def reset_acc(self):
        self.acc = dict(terms={}, n=0, vx=0.0, yaw_rate=0.0, falls=0, terms_other=0, eps=[], tmax={}, tmin={})

    def close(self):
        self.pool.close()


def train_cfg(init_std: float, desired_kl: float = 0.01, rollout_steps: int = 64, gamma: float = 0.99,
              lam: float = 0.95, value_clip: float | None = 0.2) -> dict:
    """rsl_rl's cfg. rollout_steps / gamma / lam drive the rollout and the returns for both learners; desired_kl and
    value_clip (None = unclipped; rsl_rl clips the value with clip_param, so only 0.2 or None) are the CPU learner's
    (the MLX learner takes them per update, lab/mlx_ppo.py)."""
    assert value_clip in (None, 0.2), "rsl_rl clips the value loss with clip_param (0.2): VALUE_CLIP 0.2 or None"
    return {
        "num_steps_per_env": rollout_steps, "obs_groups": {"actor": ["policy"], "critic": ["policy", "critic"]},
        "algorithm": dict(class_name="PPO", num_learning_epochs=5, num_mini_batches=4, clip_param=0.2, gamma=gamma,
                          lam=lam, value_loss_coef=1.0, entropy_coef=0.0, learning_rate=1e-3, max_grad_norm=1.0,
                          use_clipped_value_loss=value_clip is not None, schedule="fixed", desired_kl=desired_kl),
        "actor": dict(class_name="MLPModel", hidden_dims=HIDDEN, activation="elu", obs_normalization=True,
                      distribution_cfg=dict(class_name="GaussianDistribution", init_std=init_std, std_type="log")),
        "critic": dict(class_name="MLPModel", hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True),
        "multi_gpu": None,
    }


def load_mlp_into(actor, ref) -> None:
    """Copy a duck-mlp/1 (duckref DuckMlp `ref`) into the rsl_rl actor; bake + freeze its normaliser. A wider actor (a
    vision move: 61 + the eye's 12 inputs) gets the 61-input weights on its first 61 inputs, zero weights and sub 0 /
    div 1 on the rest, so it starts out acting exactly as `ref`."""
    lins = [m for m in actor.mlp if isinstance(m, nn.Linear)]
    assert len(lins) == len(ref.layers) == 4
    n_ref, n_in = len(ref.sub), lins[0].in_features
    assert n_in >= n_ref, (n_in, n_ref)
    with torch.no_grad():
        for k, (lin, (W, b, _act, _a)) in enumerate(zip(lins, ref.layers)):
            W = torch.tensor(W, dtype=torch.float32)
            if k == 0 and n_in > n_ref:
                W = torch.cat([W, torch.zeros(W.shape[0], n_in - n_ref)], 1)
            assert tuple(lin.weight.shape) == tuple(W.shape), (lin.weight.shape, W.shape)
            lin.weight.copy_(W)
            lin.bias.copy_(torch.tensor(b, dtype=torch.float32))
        pad = lambda v, fill: torch.cat([torch.tensor(v, dtype=torch.float32), torch.full((n_in - n_ref,), fill)])  # noqa: E731
        nz = actor.obs_normalizer
        nz._mean.copy_(pad(ref.sub, 0.0)[None])
        nz._std.copy_(pad(ref.div, 1.0)[None])
        nz._var.copy_(nz._std ** 2)
        nz.eps, nz.until = 0.0, 0  # (x - sub) / div exactly as the stock graph; never updated


def warm_start(actor, skill: str) -> None:
    """Copy the stock skill's duck-mlp/1 weights into the rsl_rl actor (load_mlp_into)."""
    from duckref.mlp import DuckMlp
    load_mlp_into(actor, DuckMlp(PLANT / "policies" / f"{skill}.mlp.json"))


class PlainActor(nn.Module):
    """The stock graph: Sub -> Div -> (Gemm -> Elu) x3 -> Gemm, obs[1,61] -> actions[1,14]."""

    def __init__(self, sub, div, lins):
        super().__init__()
        self.register_buffer("sub", sub.detach().cpu().reshape(-1).clone())
        self.register_buffer("div", div.detach().cpu().reshape(-1).clone())
        layers = []
        for i, lin in enumerate(lins):
            new = nn.Linear(lin.in_features, lin.out_features)
            new.load_state_dict({k: v.detach().cpu() for k, v in lin.state_dict().items()})
            layers.append(new)
            if i < len(lins) - 1:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net((obs - self.sub) / self.div)


def plain_actor(actor) -> PlainActor:
    nz = actor.obs_normalizer
    return PlainActor(nz._mean, nz._std + nz.eps, [m for m in actor.mlp if isinstance(m, nn.Linear)]).eval()


def export_onnx(pa: PlainActor, skill: str, out: Path, label: str) -> float:
    n_in = pa.sub.numel()  # 61, or a vision move's 73
    torch.onnx.export(pa, (torch.zeros(1, n_in),), str(out), input_names=["obs"], output_names=["actions"],
                      opset_version=18, dynamo=False)
    m, stock = onnx.load(str(out)), onnx.load(str(PLANT / "policies" / f"{skill}.onnx"))
    del m.metadata_props[:]
    for p in stock.metadata_props:
        m.metadata_props.add(key=p.key, value=p.value if p.key != "run_path" else label)
    onnx.checker.check_model(m)
    onnx.save(m, str(out))
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ins, outs = sess.get_inputs()[0], sess.get_outputs()[0]
    assert ins.name == "obs" and list(ins.shape) == [1, n_in] and outs.name == "actions" and list(outs.shape) == [1, 14], (ins, outs)
    x = np.random.default_rng(0).normal(size=(64, n_in)).astype(np.float32) * 0.5
    with torch.no_grad():
        ref = pa(torch.from_numpy(x)).numpy()
    got = np.concatenate([sess.run(None, {"obs": x[i:i + 1]})[0] for i in range(64)])
    return float(np.abs(got - ref).max())


def build(env: PoolVecEnv, move: Move, device: str, init_std: float, desired_kl: float = 0.01, **cfg_kw) -> PPO:
    cfg = train_cfg(init_std, desired_kl, **cfg_kw)
    with contextlib.redirect_stdout(io.StringIO()):  # rsl_rl prints the model summaries
        alg = PPO.construct_algorithm(env.get_observations(), env, cfg, device)
    try:
        warm_start(alg.actor, move.warm_start)
    except (AssertionError, RuntimeError, ValueError) as exc:
        print(f"[duck-rl] warm start skipped (shape mismatch): {exc}", flush=True)
    if os.environ.get("DUCK_WIDE_MLP"):
        from duckref.mlp import DuckMlp
        load_mlp_into(alg.actor, DuckMlp(Path(os.environ["DUCK_WIDE_MLP"])))
        print(f"[duck-rl] net2wider warm start from {os.environ['DUCK_WIDE_MLP']}", flush=True)
    if os.environ.get("DUCK_DUMP_INIT"):
        torch.save({k: v.detach().cpu().clone() for k, v in alg.actor.state_dict().items()},
                   os.environ["DUCK_DUMP_INIT"])
    if os.environ.get("DUCK_SCRATCH") == "1":
        # E3 condition B: keep the stock input normaliser (shared interface), re-draw every linear weight/bias
        # with PyTorch's default init (the same draw rsl_rl uses when nothing is warm-started).
        with torch.no_grad():
            for m in alg.actor.mlp:
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
        if os.environ.get("DUCK_DUMP_INIT_AFTER"):
            torch.save({k: v.detach().cpu().clone() for k, v in alg.actor.state_dict().items()},
                       os.environ["DUCK_DUMP_INIT_AFTER"])
    alg.actor.to(device)
    return alg


def _set_actor_trainable(alg: PPO, on: bool):  # (the loop uses overlap.Updater; mac-speed/ research scripts use this)
    for p in alg.actor.parameters():
        p.requires_grad_(on)


# ---- opt-in research options (stage 2 of the Mac-speed work; the defaults above are unchanged) -----------------------
class TwoLrAdam(torch.optim.Adam):
    """Adam with the critic in its own param group at a fixed LR: rsl_rl's adaptive schedule writes one LR into every
    group before each step, so the critic's LR is restored right before stepping. critic_lr None = the shared LR
    (train() sets it when the actor is released: the warm-up keeps rsl_rl's 1e-3, as on the MLX learner)."""

    def __init__(self, actor_params, critic_params, lr: float, critic_lr: float | None):
        super().__init__([dict(params=list(actor_params)), dict(params=list(critic_params))], lr=lr)
        self.critic_lr = critic_lr

    def step(self, closure=None):
        if self.critic_lr is not None:
            self.param_groups[1]["lr"] = self.critic_lr
        return super().step(closure)


def init_actor_from(alg: PPO, ckpt: Path, skill: str, critic: bool) -> None:
    """Continue from a saved checkpoint instead of stock: a full stage-2 checkpoint (actor incl. std; optionally the
    critic) or a plain `ckpt.pt` (weights only; std stays at init)."""
    ck = torch.load(ckpt, map_location="cpu")
    assert ck.get("warm_start", skill) == skill, (ck.get("warm_start"), skill)
    if "actor" in ck:
        alg.actor.load_state_dict(ck["actor"])
        if critic:
            alg.critic.load_state_dict(ck["critic"])
        return
    assert not critic, "a plain ckpt.pt has no critic"
    src = load_plain(ckpt)
    lins = [m for m in alg.actor.mlp if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for lin, s in zip(lins, [m for m in src.net if isinstance(m, nn.Linear)]):
            lin.weight.copy_(s.weight)
            lin.bias.copy_(s.bias)


def policy_shift(alg: PPO) -> tuple[float, float]:
    """KL(old || new) and rms change of the action mean over the rollout just used for the update (no RNG draws)."""
    st = alg.storage
    with torch.inference_mode():
        alg.actor.distribution.update(alg.actor.mlp(alg.actor.get_latent(st.observations.flatten(0, 1))))
        new = alg.actor.output_distribution_params
        old = tuple(p.flatten(0, 1) for p in st.distribution_params)
        kl = alg.actor.get_kl_divergence(old, new).mean()
        dmu = (new[0] - old[0]).pow(2).mean().sqrt()
    return float(kl), float(dmu)


def save_ckpt(alg: PPO, move: Move, path: Path, optimizer: bool = True, **meta) -> None:
    """plain (the exported actor) + actor (incl. std and its frozen normaliser) + critic (incl. its running
    normaliser) + the CPU optimizer's state (None when the MLX learner trained: its Adam state goes to the sidecar,
    opt_sidecar()); `meta` (steps, learning_rate, critic_lr, schedule, ...)."""
    with torch.random.fork_rng(devices=[]):  # PlainActor's nn.Linear init draws from the global RNG: keep it untouched
        plain = plain_actor(alg.actor).state_dict()
    meta.setdefault("learning_rate", alg.learning_rate)
    torch.save(dict(plain=plain, actor=alg.actor.state_dict(), critic=alg.critic.state_dict(),
                    optimizer=alg.optimizer.state_dict() if optimizer else None, warm_start=move.warm_start, **meta),
               path)


def opt_sidecar(ckpt: Path) -> Path:
    """Where a checkpoint's MLX Adam state lives (kit 1.6.2): ckpt-<steps>.pt -> opt-<steps>.pt beside it (any other
    name x.pt -> opt.pt). Only the run's newest checkpoint keeps one (the final one at the end): ~2.5 MB per run, not
    per checkpoint. --resume reads it; a checkpoint without one resumes with a fresh Adam."""
    ckpt = Path(ckpt)
    return ckpt.with_name("opt-" + ckpt.name[len("ckpt-"):] if ckpt.name.startswith("ckpt-") else "opt.pt")


# the update's diagnostics copied into log.jsonl (lab/mlx_ppo.py Learner.update; the CPU learner has none of them)
UPD_KEYS = ("kl_mb", "kl_final", "dmu_rms", "lr_a_mean", "lr_a_min", "lr_a_max", "lr_c_mean", "lr_c_min", "lr_c_max",
            "clip_frac", "vclip_frac", "gn_actor", "gn_critic", "w_mean", "entropy")


LR_ANNEAL_MODES = (None, "linear", "cos3")
COS3_CYCLES, COS3_FLOOR = 3, 0.1  # LR_ANNEAL "cos3": 3 equal cycles, each from LR_FIXED down to 0.1 x LR_FIXED


def lr_factor(mode: str | None, frac: float) -> tuple[float, int | None]:
    """LR_ANNEAL's factor on `lr_fixed` at the budget fraction `frac` (0..1), and the cycle index (None: no cycles).
    None: 1. "linear": 1 - frac, to 0 at the budget's end. "cos3": three equal cosine cycles over the budget with warm
    restarts (SGDR): within a cycle FLOOR + (1 - FLOOR) (1 + cos(pi x)) / 2, x = the position in the cycle (0..1), so
    the LR falls from LR_FIXED to 0.1 x LR_FIXED and restarts at LR_FIXED; the cycle index is 0, 1, 2."""
    if mode == "linear":
        return max(0.0, 1.0 - frac), None
    if mode == "cos3":
        f = min(max(float(frac), 0.0), 1.0) * COS3_CYCLES
        cycle = min(int(f), COS3_CYCLES - 1)
        return COS3_FLOOR + (1.0 - COS3_FLOOR) * 0.5 * (1.0 + float(np.cos(np.pi * (f - cycle)))), cycle
    return 1.0, None


def _num(v: float) -> float:
    """A log number: 4 significant digits below 1e-3 (LRs, KLs), else 6 decimals."""
    return float(f"{v:.4g}") if v != 0 and abs(v) < 1e-3 else round(float(v), 6)


def _opt_to_torch(d: dict) -> dict:
    """mlx_ppo opt_state() (numpy) -> tensors, so a checkpoint stays loadable with torch.load's weights_only default."""
    return {k: torch.tensor(v.astype(np.int64) if v.dtype == np.uint64 else v) for k, v in d.items()}  # (0-d stays 0-d)


def _opt_to_numpy(d: dict) -> dict:
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in d.items()}


def batch_stats(alg: PPO) -> dict:
    """The collected batch after compute_returns, before its update: the critic's explained variance of the returns
    (1 - var(R - V) / var(R), V = the behaviour critic's values), value and return means, the return's std and the
    advantage's std before rsl_rl's normalisation (R - V)."""
    st = alg.storage
    with torch.inference_mode():
        v, ret = st.values.flatten().double(), st.returns.flatten().double()
        adv = ret - v
        var_r = float(ret.var())
        return dict(ev=round(1.0 - float(adv.var()) / var_r, 5) if var_r > 0 else None, v_mean=round(float(v.mean()), 5),
                    ret_mean=round(float(ret.mean()), 5), ret_std=round(float(ret.std()), 5),
                    adv_std_raw=round(float(adv.std()), 5))


def init_value_bias(alg: PPO, obs, time_outs: list) -> float:
    """VALUE_BIAS_INIT: shift the fresh critic's output bias so its mean value on the first rollout is the mean return
    (GAE-lambda) of that rollout; the batch's values move with it, the time-out bootstraps rsl_rl already folded into
    the rewards (r += gamma V(s_t) on a time-out step; `time_outs` = that rollout's per-step masks) get the same shift,
    and the returns are recomputed: the batch is as the shifted critic would have collected it. -> the shift."""
    st = alg.storage
    with torch.no_grad():
        delta = float((st.returns - st.values).mean())
        last = [m for m in alg.critic.mlp if isinstance(m, nn.Linear)][-1]
        last.bias.add_(delta)
        st.values.add_(delta)
        for t, to in enumerate(time_outs):
            st.rewards[t].add_(alg.gamma * delta * to.to(st.rewards.dtype).reshape(st.rewards[t].shape))
    with torch.inference_mode():
        alg.compute_returns(obs)
    return delta


def _carry_fallback(e: BaseException, upd) -> None:
    """An exception leaving train() after the MLX learner fell back to the CPU one (lab/overlap.py _fall_back) carries the
    reason as `e.mlx_fallback` (the summary's mlx_fallback, which a raise never returns): PR #1 R7-2, so the caller can
    record the learner that ran even when the fallback's first log record never reached it (a checkpoint or export
    that throws in that same iteration)."""
    if upd is not None and getattr(upd, "fallback", None) is not None:
        with contextlib.suppress(AttributeError, TypeError):
            e.mlx_fallback = upd.fallback


def train(move: Move, out: Path, minutes: float, workers=8, per_worker=16, device="cpu", seed=1, init_std=0.1,
          warmup_iters=10, lr=1e-4, max_iters=100000, log_fn=print, steps=None, desired_kl=0.01, lr_fixed=None,
          critic_lr=None, init_from=None, init_critic=False, log_kl=False, env="fast", learner="cpu", overlap=False,
          iw_cap=3.0, ckpt_every_steps=0, probe=None, stop_on_probe=False, probe_pool=None,
          final_curriculum=False, rollout_steps=64, gamma=0.99, lam=0.95, value_clip=0.2, critic_epochs=None,
          std_floor=None, warmup_ev=None, value_bias_init=False, lr_anneal=None, resume=False, step_offset=0) -> dict:
    """`env` = "fast" (numba env, default) | "ref" (lab/duck.py DuckEnv; bit-identical, slower). `learner` = "cpu"
    (rsl_rl, serial; with env "ref" the pre-v2 lab bit for bit) | "mlx" (lab/overlap.py Updater; `overlap` = update
    during the next collect with decoupled PPO, importance weights capped at `iw_cap`; falls back to "cpu" on failure).
    `ckpt_every_steps` = a full checkpoint + ONNX every N env steps the weights were trained on (overlapped, the CPU
    copy is one batch behind the collected steps; checkpoints never change the training itself), taken only once the
    weights have trained in the final CURRICULUM stage; with `probe` (the move's primary probe) each is scored and the
    round ships the best one (lab/probestop.py); `stop_on_probe` = stop after 2 consecutive passes; `probe_pool` = a
    callable returning the pool to score in, called once the env workers are up (kit.py round shares that pool with
    eval). `final_curriculum` = the final stage's weights from the first step (a continued round: move.weights(1.0)).
    `steps` = env-step budget that drives the curriculum fraction (`minutes` stays a hard cap); None = the pre-v2
    minutes budget with a wall-clock curriculum. `log_fn(rec)` returning a truthy value (a str: the stopped_by) ends the
    run after that iteration, as the minutes cap does: the final checkpoint and the export follow. kit.py's defaults are fast + mlx + overlap + checkpoints every 250k +
    probe stop + a step budget per move class. The rest = the lab as used by the authors. Opt-in (stage 2):
    `desired_kl`; `lr_fixed` = fixed actor LR after warm-up; `critic_lr` = critic in its own Adam group (from the
    release of the actor; the warm-up keeps 1e-3); `init_from` = actor (and with `init_critic` the critic) from a
    checkpoint; `log_kl` = per-iteration KL / action-mean shift (CPU learner only).
    Kit 1.6.2 knobs (the defaults are the run before them, bit for bit; spikes/duck-rl/README.md "Learner knobs"):
    `rollout_steps` (64 per env per iteration), `gamma` / `lam` (0.99 / 0.95), `value_clip` (0.2; None = unclipped;
    the MLX learner takes any value), `critic_epochs` (None = 5, MLX only), `std_floor` (None), `warmup_ev` (None =
    a fixed `warmup_iters`; else the actor is released once the last batch's explained variance >= it, at the latest at
    `warmup_iters`), `value_bias_init` (the fresh critic's output bias = the first rollout's mean return), `lr_anneal`
    ("linear": `lr_fixed` and `critic_lr` fall linearly to 0 at the budget's end; "cos3": the actor's `lr_fixed` runs
    three cosine cycles down to 0.1x with warm restarts (lr_factor), the critic exempt at a constant `critic_lr`, else
    `lr_fixed`; each record carries `lr_cycle`), `resume` (with `init_from`: also the
    optimizer state (MLX: the checkpoint's sidecar opt-<steps>.pt, kept for the run's newest and final checkpoint only;
    CPU: `optimizer`; none: a fresh Adam, with a note) and the LR, and no critic warm-up when a critic is loaded).
    `step_offset` (the duck-rl kit's --resume, 1.9.3): the run's steps before this session; checkpoint names and their
    `steps` continue the run's count (ckpt-<offset + trained>), `session_steps` keeps this session's; the records,
    the curriculum and the budget count this session's steps.
    The per-iteration record also carries the update's diagnostics (labelled with `upd_it`, the iteration whose batch
    the update trained on: it - 1 when overlapped) and the batch's own stats (ev, v_mean, ret_*, adv_std_raw)."""
    assert learner == "cpu" or not log_kl, "log_kl needs the CPU learner"
    assert learner == "mlx" or (critic_epochs is None and value_clip in (None, 0.2)), \
        "critic_epochs and a value clip other than 0.2 / None need the MLX learner"
    assert lr_anneal in LR_ANNEAL_MODES and (lr_anneal is None or lr_fixed is not None), \
        f"lr_anneal {lr_anneal!r}: one of {LR_ANNEAL_MODES}, and a mode needs lr_fixed"
    if lr_anneal == "cos3" and critic_lr is None:
        critic_lr = lr_fixed  # cos3: the critic is exempt from the cycles (its own Adam group at a constant LR)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    t_setup = time.time()
    env_kind, env, upd, ps, log = env, None, None, None, None
    ckpts, ckpt_s = {}, [0.0]  # trained steps -> ORT vs torch of its ONNX; time spent saving + exporting
    opt_last = [None]  # the newest checkpoint's Adam-state sidecar (the only one kept)
    last_start = max((start for start, _ in move.curriculum), default=0.0)
    final_from = 0 if final_curriculum or last_start == 0 else None  # trained steps from which the weights have seen
    #                                                                   final-stage batches (checkpoints start there)

    def checkpoint(trained: int, iters: int, final: bool = False) -> bool | None:
        """Checkpoint of the CPU copy (RNG-neutral) + its ONNX, scored on the probe -> passed (None: no probe). The
        final weights of a run that ended on its budget are scored with the stop rule off."""
        nonlocal stopped_by
        run_steps = step_offset + trained  # (the run's count: --resume continues it)
        t, pt = time.time(), out / f"ckpt-{run_steps:07d}.pt"
        save_ckpt(alg, move, pt, optimizer=upd.name == "cpu", iters=iters, steps=run_steps,
                  **({"session_steps": trained, "step_offset": step_offset} if step_offset else {}),
                  elapsed_s=round(time.time() - t0, 1), learning_rate=upd.learning_rate, critic_lr=upd.critic_lr,
                  schedule=upd.schedule, released=released)
        mlx_opt = upd.opt_state() if upd.name == "mlx" else None  # (the Adam state of these weights: --resume)
        if mlx_opt:  # the sidecar of the newest checkpoint only: the previous one goes (opt_sidecar)
            op = opt_sidecar(pt)
            torch.save(dict(mlx_opt=_opt_to_torch(mlx_opt), steps=run_steps, ckpt=pt.name), op)
            if opt_last[0] is not None and opt_last[0] != op:
                opt_last[0].unlink(missing_ok=True)
            opt_last[0] = op
        with torch.random.fork_rng(devices=[]):
            ckpts[trained] = export_onnx(plain_actor(alg.actor), move.warm_start, pt.with_suffix(".onnx"),
                                         f"move-lab {move.slug} {pt.stem}")
        ckpt_s[0] += time.time() - t
        if ps is None:
            return None
        passed, stop = ps.score(trained, pt, pt.with_suffix(".onnx"), may_stop=not final)
        stopped_by = "probe" if stop else stopped_by
        return passed

    # the MLX update's non-default knobs (absent = mlx_ppo's constants: the pre-1.6.2 update exactly)
    knobs = {k: v for k, v in (("dkl", desired_kl), ("value_clip", value_clip), ("critic_epochs", critic_epochs),
                               ("std_floor", std_floor)) if v != {"dkl": 0.01, "value_clip": 0.2}.get(k)}
    released, warmup_end, last_ev, prev_bst, bias_shift, opt_state = False, None, None, None, None, None
    try:
        env = PoolVecEnv(move, workers, per_worker, seed, env_kind)
        alg = build(env, move, device, init_std, desired_kl, rollout_steps=rollout_steps, gamma=gamma, lam=lam,
                    value_clip=value_clip)
        if critic_lr is not None:  # (the critic's own LR from the release on: Updater.release)
            alg.optimizer = TwoLrAdam(alg.actor.parameters(), alg.critic.parameters(), alg.learning_rate, None)
        if init_from:
            init_actor_from(alg, Path(init_from), move.warm_start, init_critic)
        if resume:  # --resume: the optimizer state, the LR, no critic warm-up when a critic was loaded
            ck = torch.load(init_from, map_location="cpu")
            side = opt_sidecar(Path(init_from))
            sd = torch.load(side, map_location="cpu") if side.exists() else {}
            if sd and sd.get("steps") != ck.get("steps"):
                sd = {}  # (a sidecar of other weights)
            notes = []
            if learner == "mlx" and (sd.get("mlx_opt") or ck.get("mlx_opt")):
                opt_state = _opt_to_numpy(sd.get("mlx_opt") or ck["mlx_opt"])
                notes.append(f"the MLX Adam states (actor + critic) from {side.name if sd else 'the checkpoint'}")
            elif learner == "cpu" and ck.get("optimizer"):
                try:
                    alg.optimizer.load_state_dict(ck["optimizer"])
                    notes.append("the CPU Adam state")
                except (ValueError, KeyError) as e:
                    notes.append(f"NOT the CPU Adam state ({e})")
            elif learner == "cpu" and (sd.get("mlx_opt") or ck.get("mlx_opt")):
                # an MLX run's checkpoint resumed on the CPU learner: its optimiser state is there, but in MLX's form
                notes.append(f"MLX optimiser state ({side.name if sd.get('mlx_opt') else 'in the checkpoint'}) is not usable "
                             "by the CPU learner: Adam starts fresh")
            else:
                notes.append(f"no {learner.upper()} optimizer state for it (no {side.name}: only a run's newest or "
                             "final checkpoint keeps one, kit >= 1.6.2): Adam starts fresh")
            if lr_fixed is None and ck.get("learning_rate"):
                lr = float(ck["learning_rate"])
                notes.append(f"the adaptive LR {lr:.3g}")
            if init_critic:
                warmup_iters, warmup_ev = 0, None
                notes.append("the critic (no warm-up)")
            else:
                notes.append("no critic in it: the critic warm-up runs")
            print(f"[resume] {init_from}: " + "; ".join(notes), flush=True)
        alg.train_mode()
        # the env's own act (the duck-rl kit's fast rollout, duck_rl/rollout.py: the same numbers), else PPO.act
        act = env.rollout_act(alg, rollout_steps) if hasattr(env, "rollout_act") else alg.act
        upd = Updater(alg, learner, overlap, seed, iw_cap, knobs=knobs, opt_state=opt_state)  # (actor frozen: warm-up)
        fallback_logged = False  # whether a record carried upd.fallback yet (mlx_fallback below)
        ps = ProbeStop(move, probe, stop_on_probe, probe_pool and probe_pool()) if ckpt_every_steps and probe else None
        obs = env.get_observations().to(device)
        log = open(out / "log.jsonl", "w")
        budget, t0, total, it, next_ckpt, stopped_by = minutes * 60.0, time.time(), 0, 0, ckpt_every_steps, None
        setup_s = t0 - t_setup
        while it < max_iters and stopped_by is None:
            if steps:
                frac = total / steps
                if frac >= 1.0 or time.time() - t0 >= budget:
                    stopped_by = "steps" if frac >= 1.0 else "minutes"
                    break
            else:
                frac = (time.time() - t0) / budget
                if frac >= 1.0:
                    stopped_by = "minutes"
                    break
            cfrac = 1.0 if final_curriculum else frac  # (the curriculum's fraction)
            env.weights = move.weights(cfrac)
            anneal, lr_cycle = lr_factor(lr_anneal, frac)
            c_anneal = 1.0 if lr_anneal == "cos3" else anneal  # (cos3: the critic is exempt)
            if not released and (it >= warmup_iters or (warmup_ev is not None and last_ev is not None
                                                        and last_ev >= warmup_ev)):
                # critic warmed up: release the actor at a small adaptive LR (or LR_FIXED)
                upd.release(*(("adaptive", lr) if lr_fixed is None else ("fixed", lr_fixed * anneal)),
                            critic_lr * c_anneal if critic_lr is not None else None)
                released, warmup_end = True, it
            elif released and lr_anneal is not None:  # LR_ANNEAL: linear to 0 at the budget's end, or cos3's cycles
                upd.set_lr("fixed", lr_fixed * anneal, critic_lr * c_anneal if critic_lr is not None else None)
            env.reset_acc()
            ts = time.time()
            bias_now = it == 0 and value_bias_init and not init_critic  # VALUE_BIAS_INIT: before the first update
            tos = []  # (its rollout's time-out masks)
            with torch.inference_mode():
                for _ in range(rollout_steps):
                    actions = act(obs)
                    obs, rew, dones, extras = env.step(actions)
                    if bias_now:
                        tos.append(extras["time_outs"].clone())
                    obs, rew, dones = obs.to(device), rew.to(device), dones.to(device)
                    alg.process_env_step(obs, rew, dones, extras)
                t_col = time.time() - ts
                alg.compute_returns(obs)
            if bias_now:
                bias_shift = init_value_bias(alg, obs, tos)
                upd.load_params()
            bst = batch_stats(alg)  # this batch, before its update
            last_ev = bst["ev"]
            a, n = env.acc, env.acc["n"]
            # a checkpoint will follow this update (the condition below): take the MLX Adam state between the wait and
            # the submit, where it matches the CPU copy's weights (no extra wait)
            fetch = bool(ckpt_every_steps) and upd.name == "mlx" and total + (0 if upd.overlap else n) >= next_ckpt
            losses, t_upd, t_wait = upd.update(fetch_opt=fetch)  # overlapped: the previous batch's update
            t_iter = time.time() - ts
            lbst, l_it = (prev_bst, it - 1) if upd.busy else (bst, it)  # the batch the returned update trained on
            prev_bst = bst
            total += n
            eps = a["eps"]
            rec = dict(it=it, steps=total, elapsed_s=round(time.time() - t0, 1), sps=round(n / t_iter),
                       sps_collect=round(n / t_col), t_collect=round(t_col, 3), t_update=round(t_upd, 3),
                       t_wait=round(t_wait, 3), learner=upd.name, warmup=not released,
                       stage=move.stage(cfrac),
                       weights={k: round(v, 3) for k, v in env.weights.items()},
                       terms={k: round(v / n, 5) for k, v in a["terms"].items()},
                       reward_step=round(sum(env.weights[k] * v / n for k, v in a["terms"].items()), 5),
                       vx=round(a["vx"] / n, 4), abs_yaw_rate=round(a["yaw_rate"] / n, 4),
                       falls_per_1k=round(1000 * a["falls"] / n, 3), other_terms_per_1k=round(1000 * a["terms_other"] / n, 3),
                       ep_n=len(eps), ep_len=round(float(np.mean([e["len"] for e in eps])), 1) if eps else None,
                       ep_fall_frac=round(float(np.mean([e["fell"] for e in eps])), 3) if eps else None,
                       term_max={k: round(v, 5) for k, v in a["tmax"].items()}, term_min={k: round(v, 5) for k, v in a["tmin"].items()},
                       std=round(float(alg.actor.output_std.detach().mean()), 4), lr=upd.learning_rate,
                       v_loss=round(losses["value"], 5) if losses else None,
                       surrogate=round(losses["surrogate"], 5) if losses else None,
                       **{k: round(losses[k], 5) for k in ("w_max", "iw_capped", "ratio_clamped") if losses and k in losses})
            # kit 1.6.2 diagnostics: the update's (of batch `upd_it`: it - 1 when overlapped) and the run's state
            ls = alg.actor.distribution.log_std_param.detach().exp()
            rec.update(std_min=round(float(ls.min()), 5), std_max=round(float(ls.max()), 5),
                       loadavg1=round(os.getloadavg()[0], 2))
            if lr_cycle is not None:  # LR_ANNEAL "cos3": the cycle of the LR set this iteration (the record's `lr`)
                rec["lr_cycle"] = lr_cycle
            if losses:
                # upd_warmup / upd_lr: that update's own warm-up flag and end LR (the record's `warmup` and `lr` describe
                # the update submitted this iteration, one ahead when overlapped); the CPU learner is serial: the same
                rec["upd_it"] = l_it
                rec["upd_warmup"] = not losses["actor_trained"] if "actor_trained" in losses else not released
                rec["upd_lr"] = losses.get("lr_end", upd.learning_rate)
                rec.update({k: _num(losses[k]) for k in UPD_KEYS if k in losses})
                if lbst:
                    rec.update(lbst)
            if it == 0 and bias_shift is not None:
                rec["value_bias_shift"] = round(bias_shift, 5)
            if upd.fallback is not None and not fallback_logged:
                # the first record after the MLX learner fell back to the CPU one: its reason (the summary's mlx_fallback),
                # so a caller sees it even when the run ends without a summary (a crash, a stop)
                rec["mlx_fallback"] = upd.fallback
                fallback_logged = True
            if log_kl:
                rec["kl"], rec["dmu_rms"] = (round(v, 6) for v in policy_shift(alg))
            if final_from is None and cfrac >= last_start:  # this batch is the first one of the final stage
                final_from = total
            trained, n_trained = total - (n if upd.busy else 0), it + (0 if upd.busy else 1)  # in the CPU copy
            if ckpt_every_steps and trained >= next_ckpt:
                if final_from is not None and trained >= final_from:
                    rec["ckpt"], rec["ckpt_pass"] = trained, checkpoint(trained, n_trained)
                next_ckpt = (trained // ckpt_every_steps + 1) * ckpt_every_steps
            log.write(json.dumps(rec) + "\n")
            log.flush()
            stop = log_fn(rec)  # a truthy return stops the run here, as the minutes cap does (a deadline of the caller's)
            stopped_by = (stop if isinstance(stop, str) else "log_fn") if stop else stopped_by
            it += 1
        upd.finish()  # overlapped: the last update -> the CPU copy, before the export
        if (ckpt_every_steps and stopped_by != "probe" and final_from is not None and total >= final_from
                and (not ckpts or max(ckpts) < total)):  # the final weights, with the stop rule off (not in log.jsonl)
            log_fn(dict(ckpt=total, ckpt_pass=checkpoint(total, it, final=True), final=True))
        shipped = ps.select() if ps else None
        if ps:
            ps.write(out / "probe_ckpts.json")
    except BaseException as e:
        _carry_fallback(e, upd)
        raise
    finally:
        for closer in (upd, env, log, ps):
            if closer is not None:
                closer.close()
    try:
        if shipped:  # the round's policy = the selected checkpoint (L2)
            shutil.copy(shipped["pt"], out / "ckpt.pt")
            shutil.copy(shipped["onnx"], out / "policy.onnx")
            err = ckpts[shipped["steps"]]
        else:
            pa = plain_actor(alg.actor)
            torch.save(dict(plain=pa.state_dict(), warm_start=move.warm_start, iters=it, steps=total), out / "ckpt.pt")
            err = export_onnx(pa, move.warm_start, out / "policy.onnx", f"move-lab {move.slug}")
    except BaseException as e:
        _carry_fallback(e, upd)
        raise
    summary = dict(iters=it, steps=total, train_s=round(time.time() - t0, 1), setup_s=round(setup_s, 1),
                   onnx=str(out / "policy.onnx"), onnx_vs_torch=err, device=device, workers=workers, per_worker=per_worker,
                   env=env_kind, learner=learner, overlap=overlap, iw_cap=iw_cap if overlap else None,
                   mlx_fallback=upd.fallback, stopped_by=stopped_by, init=str(init_from) if init_from else "stock",
                   rollout_steps=rollout_steps, knobs=knobs or None, released_at_it=warmup_end, resume=bool(resume),
                   **({"value_bias_shift": round(bias_shift, 5)} if bias_shift is not None else {}))
    if ps:  # what the author sees of the probe: no per-checkpoint numbers (lab/probestop.py)
        summary.update(ckpt_every_steps=ckpt_every_steps, checkpoints=len(ckpts), ckpt_s=round(ckpt_s[0], 1),
                       final_stage_from=final_from, stop_step=ps.stop_step if stopped_by == "probe" else None,
                       passed_ckpts=[c["steps"] for c in ps.ckpts if c.get("passed")],
                       shipped=dict(steps=shipped["steps"], pass40=shipped["pass40"], falls40=shipped["falls40"],
                                    n=40) if shipped else None,
                       probe_s=round(ps.blocked_s, 1))
    if final_curriculum:
        summary["curriculum"] = "final stage from the first step"
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def load_plain(ckpt: Path) -> PlainActor:
    sd = torch.load(ckpt, map_location="cpu")["plain"]
    lins = []
    for i in range(0, 7, 2):
        W = sd[f"net.{i}.weight"]
        lin = nn.Linear(W.shape[1], W.shape[0])
        lin.load_state_dict({"weight": W, "bias": sd[f"net.{i}.bias"]})
        lins.append(lin)
    return PlainActor(sd["sub"], sd["div"], lins).eval()
