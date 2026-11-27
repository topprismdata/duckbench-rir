"""L2 (kit v2): score checkpoints on the probe during training, stop early, ship the best checkpoint.

learn.train saves a checkpoint (and its ONNX) every `ckpt_every_steps` env steps, counted by the steps the weights were
trained on. ProbeStop scores it on the move's primary probe over the 8 official public seeds, blocking, on the replay
path (lab.replay.episode: duckref reference loop + onnxruntime) in a pool of 8 processes started once per run. The run
stops after 2 consecutive passing checkpoints (or at the step budget). Selection: every passing checkpoint plus the
last two are scored on 40 public episodes (public_episode(0..39), incl. the official 8) after training; the pick is
the one that meets the verdict (>= 30 passed, <= 3 falls), then the most passes, then the fewest falls, then the
earlier checkpoint. The pick and eval's 40-episode verdict use the same public episodes, so that verdict is optimistic;
the hidden-seed count (eval) is the unbiased check. (All probe work blocks training: running it alongside, even the
40-episode extension alone, slowed the lock-step collect 2-5x for several iterations, M.)
Hidden seeds are never used here: kit.py eval scores them on the shipped policy only. The author sees the stop step,
which checkpoints passed, the shipped checkpoint and its 40-episode count; the per-checkpoint probe numbers stay in
the run dir (probe_ckpts.json), so nobody tunes against a probe curve.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

from .replay import N_PUBLIC, episode, public_episode, traj_of

N_EXT, GOOD_PASS, GOOD_FALLS = 40, 30, 3
PASSES_TO_STOP = 2
TIMEOUT_S = 300  # for one batch of probe episodes (8-64 of them take 0.3-7 s, M): a hung worker raises, not waits

def _warm(policy: str):
    """Pool workers warm up (imports, onnxruntime, the first plant) while training starts: the first checkpoint's
    scoring took 1.5 s cold vs 0.7 s warm (M)."""
    episode(dict(policy=policy, onnx=None, cmd=[0.0] * 13, ep=public_episode(0), ticks=10))


def probe_pool(policy: str, procs: int = 8):
    """The replay-path process pool (probe episodes, eval clips), warmed up; kit.py round shares one across train
    and eval."""
    return mp.get_context("spawn").Pool(procs, initializer=_warm, initargs=(policy,))


class ProbeStop:
    def __init__(self, move, primary: str, stop: bool, pool=None):
        from probes import PROBES
        from probes.common import SEED_PASS_FRAC
        self.move, self.primary, self.stop = move, primary, stop
        self.probe, self.need = PROBES[primary], SEED_PASS_FRAC * N_PUBLIC
        self.own_pool = pool is None
        self.pool = probe_pool(move.warm_start) if pool is None else pool
        self.ckpts: list[dict] = []
        self.blocked_s = self.select_s = 0.0
        self.stop_step = None  # the second of 2 consecutive passing checkpoints

    def _run(self, onnx: str, eps: range):
        """Episodes public_episode(i), i in eps, on the replay path, in the pool -> AsyncResult."""
        return self.pool.map_async(episode, [dict(policy=self.move.warm_start, onnx=onnx, cmd=self.move.eval_cmd.tolist(),
                                                  ep=public_episode(i)) for i in eps])

    def _eval(self, res) -> list[tuple[bool, bool]]:
        """AsyncResult of episodes -> per episode (passed, fell) on the primary probe (waits until done)."""
        per = [self.probe.evaluate(traj_of(r)) for r in res.get(timeout=TIMEOUT_S)]
        return [(bool(p["passed"]), not p["no_fall"]) for p in per]

    def score(self, steps: int, pt: Path, onnx: Path, may_stop: bool = True) -> tuple[bool, bool]:
        """One checkpoint on the 8 official seeds, blocking -> (passed, stop now). may_stop=False: a run's final
        weights, scored for the selection only."""
        t = time.time()
        c = dict(steps=steps, pt=str(pt), onnx=str(onnx), seeds8=self._eval(self._run(str(onnx), range(N_PUBLIC))))
        c["passed"] = sum(p for p, _ in c["seeds8"]) >= self.need
        self.ckpts.append(c)
        last = self.ckpts[-PASSES_TO_STOP:]
        if (may_stop and self.stop and self.stop_step is None and len(last) == PASSES_TO_STOP
                and all(x["passed"] for x in last)):
            self.stop_step = steps
        c["t_s"] = time.time() - t
        self.blocked_s += c["t_s"]
        return c["passed"], self.stop_step is not None

    def select(self) -> dict | None:
        """Every passing checkpoint + the last two on 40 public episodes; the highest pass count, ties -> earlier."""
        t = time.time()
        seen, cands = set(), []
        for c in [c for c in self.ckpts if c["passed"]] + self.ckpts[-2:]:
            if c["steps"] not in seen:
                seen.add(c["steps"])
                cands.append(c)
        ext = [self._run(c["onnx"], range(N_PUBLIC, N_EXT)) for c in cands]  # (one queue for all candidates)
        for c, res in zip(cands, ext):
            seeds = c["seeds8"] + self._eval(res)
            c["pass40"], c["falls40"] = sum(p for p, _ in seeds), sum(f for _, f in seeds)
        self.select_s = time.time() - t
        self.blocked_s += self.select_s
        good = lambda c: c["pass40"] >= GOOD_PASS and c["falls40"] <= GOOD_FALLS  # noqa: E731
        return max(cands, key=lambda c: (good(c), c["pass40"], -c["falls40"], -c["steps"])) if cands else None

    def write(self, path: Path) -> None:
        """Per-checkpoint probe results for the maintainer (the run dir is off-limits to authors)."""
        rows = [dict(steps=c["steps"], official=f'{sum(p for p, _ in c["seeds8"])}/{N_PUBLIC}',
                     falls8=sum(f for _, f in c["seeds8"]), passed=c["passed"], t_s=round(c["t_s"], 2),
                     **({"pass40": c["pass40"], "falls40": c["falls40"]} if "pass40" in c else {})) for c in self.ckpts]
        path.write_text(json.dumps(dict(probe=self.primary, rows=rows, blocked_s=round(self.blocked_s, 2),
                                        select_s=round(self.select_s, 2)), indent=1))

    def close(self) -> None:
        if self.own_pool:
            self.pool.terminate()
            self.pool.join()
