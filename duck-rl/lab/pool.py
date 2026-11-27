"""Multiprocess CPU env pool: W worker processes x K DuckEnvs, stepped in lock-step, auto-reset inside the worker.

step(actions[N,14]) -> obs[N,61], priv[N,6], state[N,D] (the post-step state, before any auto-reset), terminated[N],
timeout[N], episodes (list of finished-episode dicts). obs/priv are post-reset for envs that just finished.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np


def _worker(remote, cfg, seeds):
    import os
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(k, "1")
    from .duck import DuckEnv
    envs = [DuckEnv(cfg["policy"], cfg["lo"], cfg["hi"], cfg["episode_s"], cfg["terminations"], s) for s in seeds]
    first = [e.reset() for e in envs]
    stats = [dict(ret_fell=0) for _ in envs]
    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            remote.send((np.stack([o for o, _ in first]), np.stack([p for _, p in first])))
        elif cmd == "step":
            n = len(envs)
            obs = np.empty((n, 61), np.float32)
            priv = np.empty((n, envs[0].PRIV), np.float32)
            S, term, tout, eps = None, np.zeros(n, bool), np.zeros(n, bool), []
            for i, (e, a) in enumerate(zip(envs, data)):
                o, p, s, tm, to = e.step(a)
                if S is None:
                    S = np.empty((n, s.shape[0]), np.float32)
                S[i], term[i], tout[i] = s, tm, to
                if tm or to:
                    eps.append(dict(len=e.k, fell=float(s[-3] > 0), terminated=float(tm), disp=float(np.hypot(*e.tracker.prev["disp"]))))
                    o, p = e.reset()
                obs[i], priv[i] = o, p
            remote.send((obs, priv, S, term, tout, eps, np.array([e.k for e in envs])))
        elif cmd == "close":
            remote.close()
            return


class Pool:
    def __init__(self, cfg: dict, workers: int = 8, per_worker: int = 16, seed: int = 0):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for w in range(workers):
            a, b = ctx.Pipe()
            seeds = [seed * 100_000 + w * 1000 + i for i in range(per_worker)]
            p = ctx.Process(target=_worker, args=(b, cfg, seeds), daemon=True)
            p.start()
            self.remotes.append(a)
            self.procs.append(p)
        self.per, self.n = per_worker, workers * per_worker

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        res = [r.recv() for r in self.remotes]
        return np.concatenate([o for o, _ in res]), np.concatenate([p for _, p in res])

    def step(self, actions: np.ndarray):
        for i, r in enumerate(self.remotes):
            r.send(("step", actions[i * self.per:(i + 1) * self.per]))
        res = [r.recv() for r in self.remotes]
        cat = lambda k: np.concatenate([x[k] for x in res])  # noqa: E731
        return cat(0), cat(1), cat(2), cat(3), cat(4), [e for x in res for e in x[5]], cat(6)

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=2)
