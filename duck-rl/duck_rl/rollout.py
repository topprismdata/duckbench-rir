"""The fast rollout (kit 1.8.0; spikes/physics-speedup/REPORT.md, fast/README.md): the same training, bit for bit, in
less wall-clock. The default rollout path of train.py (ROLLOUT "fast"; "stock" = the kit 1.7 path). Three levers:

  fused  each env runs a tick's 4 physics substeps in one nogil numba call (duck_rl/fused.py), the BAM and mj_step
         with no Python in between (solo Challenges and duels alike); where the kernel is not available (no pinned mjData
         offsets for this mujoco, no libmujoco symbol, no numba) the envs run the stock loop;
  pipe   the trainer and the env workers overlap (Pipeliner below + env.ChallengePool "steph"): each worker's envs are two
         halves; as soon as half A's results of step t arrive, the trainer acts on them and sends A's step t+1 while
         the workers still step B; then it acts on B and sends B's t+1 before the reward and the storage of step t;
  big    4 MB socket buffers on the worker pipes (env.bigbuf; macOS' 8 KB would block a worker on its own reply).

Why it is bit-identical: every env's action still comes from its own latest obs and the current weights (the
pipeline drains at each rollout's end, so the next rollout's first actions come from the updated weights); the
action noise is drawn as rsl_rl's PPO.act draws it, one standard-normal [N, 14] tensor per step from torch's default
generator (Normal.sample = torch.normal(loc, scale) = normal_(0, 1) * scale + loc), each half taking its rows; a
half's actor/critic forward equals those rows of the full-batch forward (row-independent GEMMs; checked on the M2 at
1 and 4 threads). The observation normalisers only change in the update, between rollouts. tools/rollout_identity.py
checks it (the per-tick parity of the fused envs, and the rollout-storage hashes of a short training, stock vs fast).

Measured (M2 Air, 4 workers x 32 envs, MLX learner; the README's "Fast rollout"): x1.09-1.20 on Sprint and x1.13-1.25
on Saucer on a busy Mac (the spike: x1.27 on both). The gain needs spare CPU (the pipeline keeps 5 threads busy on 4
performance cores): on a loaded Mac it shrinks.
"""
from __future__ import annotations

import numpy as np

MODES = ("fast", "stock")


def options(mode: str) -> dict:
    """The pool options of a rollout mode (env.ChallengePool keyword arguments)."""
    if mode not in MODES:
        raise ValueError(f"ROLLOUT {mode!r}: {' or '.join(map(repr, MODES))}")
    fast = mode == "fast"
    return dict(fused=fast, pipe=fast, big=fast)


STOCK = "stock (lock-step pool, the stock substep loop)"


def describe(mode: str, per_worker: int) -> str:
    """The trainer's start line: what the rollout will try here (ran() says what ran)."""
    if mode == "stock":
        return STOCK
    from .fused import available
    ok, why = available()
    parts = ["fused substeps" if ok else f"stock substeps (the fused kernel is off: {why})",
             "pipelined halves" if per_worker >= 2 else "lock-step (1 env per worker: no halves to pipeline)",
             "4 MB pipe buffers"]
    return "fast (requested): " + ", ".join(parts)


def ran(mode: str, pool) -> str:
    """What the rollout actually ran, from the pool (env.ChallengePool) after training: the envs the workers last reported
    on fused substeps (an env stays on the stock loop until its offset check passes, and for good if it fails or the
    kernel cannot compile), the pipeline and the buffers, e.g. "fast: fused 128/128 envs, pipelined halves, 4 MB
    buffers ok" (stats.json `rollout`)."""
    if mode == "stock":
        return STOCK
    from .fused import available
    nf = getattr(pool, "fused_envs", None)
    ok, why = available()
    fused = f"fused {nf}/{pool.n} envs" if nf is not None else "fused ?/{} envs (no step reported)".format(pool.n)
    if nf is not None and nf < pool.n and not ok:
        fused += f" ({why})"
    pipe = "pipelined halves" if pool.pipe else f"lock-step ({pool.pipe_off})"
    buf = "4 MB buffers ok" if pool.big_ok else "4 MB buffers failed"
    return f"fast: {fused}, {pipe}, {buf}"


def make_act(alg, pool, rollout_steps: int):
    """The act function of the training loop (lab/learn.py): the Pipeliner's when the pool pipelines (env.ChallengePool
    pipe), else rsl_rl's alg.act."""
    if not getattr(pool, "pipe", False):
        return alg.act
    return Pipeliner(alg, pool, rollout_steps).act


class Pipeliner:
    """rsl_rl PPO.act, half by half, so the workers never wait for the trainer inside a rollout (the module doc). Fills
    alg.transition exactly as PPO.act does (rsl-rl-lib 5.5.1, a Gaussian actor head); the pool calls prefetch() as
    each half's results arrive (pool.hook)."""

    def __init__(self, alg, pool, rollout_steps: int):
        import torch
        self.torch, self.alg, self.pool, self.T = torch, alg, pool, int(rollout_steps)
        if not hasattr(alg.actor.distribution, "_distribution"):
            raise TypeError("the fast rollout needs rsl_rl's Gaussian actor head (actor.distribution._distribution)")
        self.t, self.pre = 0, None
        self.ti = [torch.from_numpy(i) for i in pool.idx]
        pool.hook = self.prefetch

    def _half(self, po, pr, z_h) -> dict:
        from tensordict import TensorDict
        td = TensorDict({"policy": po, "critic": pr}, batch_size=[po.shape[0]])
        actor = self.alg.actor
        mean = actor(td)  # the MLP output (the Gaussian head's deterministic output)
        actor.distribution.update(mean)
        d = actor.distribution._distribution
        act = (z_h * d.scale) + d.loc  # = torch.normal(loc, scale), z_h its normal_(0, 1) draw
        return dict(actions=act, values=self.alg.critic(td), logp=actor.get_output_log_prob(act),
                    params=tuple(actor.output_distribution_params), mean=mean, obs=po)

    def _draw(self):
        return self.torch.empty(self.pool.n, 14).normal_(0.0, 1.0)  # the step's noise, in step order as PPO.act draws it

    def prefetch(self, h: int, obs_h: np.ndarray, priv_h: np.ndarray) -> None:
        """Half h's results of this step arrived: act on them and send its next step (not after the rollout's last)."""
        if self.t >= self.T:  # the rollout's last step: the update comes next (the next rollout starts afresh)
            if h == 1:
                self.t = 0
            return
        torch = self.torch
        if h == 0:
            z = self._draw()
            A = self._half(torch.from_numpy(obs_h), torch.from_numpy(priv_h), z[self.ti[0]])
            self.pool.send_half(0, A["actions"].numpy())
            self.pre = [z, A, None]
        else:
            z = self.pre[0]
            B = self._half(torch.from_numpy(obs_h), torch.from_numpy(priv_h), z[self.ti[1]])
            self.pool.send_half(1, B["actions"].numpy())
            self.pre[2] = B

    def act(self, obs):
        torch, alg, ia, ib = self.torch, self.alg, self.ti[0], self.ti[1]
        po, pr = obs["policy"], obs["critic"]
        if self.pre is None:  # the rollout's first step
            z = self._draw()
            A = self._half(po[ia], pr[ia], z[ia])
            self.pool.send_half(0, A["actions"].numpy())
            B = self._half(po[ib], pr[ib], z[ib])
            self.pool.send_half(1, B["actions"].numpy())
        else:
            (z, A, B), self.pre = self.pre, None
            if not (torch.equal(A["obs"], po[ia]) and torch.equal(B["obs"], po[ib])):
                raise RuntimeError("fast rollout: the prefetched observations differ from the step's (ROLLOUT 'stock' "
                                   "runs the stock path)")

        def full(a, b):
            out = torch.empty((self.pool.n,) + tuple(a.shape[1:]), dtype=a.dtype)
            out[ia], out[ib] = a, b
            return out
        tr = alg.transition
        tr.hidden_states = (alg.actor.get_hidden_state(), alg.critic.get_hidden_state())
        tr.actions = full(A["actions"], B["actions"])
        tr.values = full(A["values"], B["values"])
        tr.actions_log_prob = full(A["logp"], B["logp"])
        tr.distribution_params = tuple(full(pa, pb) for pa, pb in zip(A["params"], B["params"]))
        tr.observations = obs
        alg.actor.distribution.update(full(A["mean"], B["mean"]))  # leave the head's state as PPO.act does
        self.t += 1
        return tr.actions
