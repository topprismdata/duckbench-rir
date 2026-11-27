"""The replay path (probe authority): an ONNX in the plain duckref plant-v1 loop (DuckRefBam.tick, reference BAM,
onnxruntime) under the same Dare randomisation and pushes as training. Public and hidden probe seeds live here."""
from __future__ import annotations

import hashlib
import hmac
import multiprocessing as mp
import os
from pathlib import Path

import mujoco
import numpy as np

from . import HOME, PLANT
from .duck import (DT, PUSH_PROBE, DuckEnv, Tracker, apply_episode, apply_push, draw_episode, is_fall, make_duck,
                   plant_model, unpack)

EPISODE_S = 10.0
N_PUBLIC, N_HIDDEN = 8, 16


def public_episode(i: int) -> dict:
    return draw_episode(np.random.default_rng([0x5EED, 0xD0C4, i]), EPISODE_S, PUSH_PROBE)


def hidden_episode(i: int) -> dict:
    key = HOME / "hidden.key"
    if not key.exists():
        HOME.mkdir(parents=True, exist_ok=True)
        key.write_bytes(os.urandom(32))
    h = hmac.new(key.read_bytes(), f"move-lab/hidden/{i}".encode(), hashlib.sha256).digest()
    return draw_episode(np.random.default_rng(int.from_bytes(h[:16], "little")), EPISODE_S, PUSH_PROBE)


class ZeroSession:
    def run(self, _names, _feeds):
        return [np.zeros((1, 14), np.float32)]


def session(onnx):
    import onnxruntime as ort
    if onnx == "zero":
        return ZeroSession()
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    return ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])


def episode(job: dict) -> dict:
    """job: policy (obs layout / stock skill), onnx (path | 'zero' | None = the stock skill), cmd (13), ep, ticks,
    bam ('ref' | 'numba'), stop_on_fall, reparse (parse the XML instead of copying the compiled model; tests).
    Returns the stacked duck-state array and the fall flag."""
    duck = make_duck(job.get("bam", "ref"), model=None if job.get("reparse") else plant_model())
    pol = job["policy"]
    if job.get("onnx"):
        duck._sessions[pol] = session(job["onnx"])
    ep, cmd = job["ep"], np.asarray(job["cmd"], float)
    apply_episode(duck, ep)
    tr = Tracker(duck, EPISODE_S)
    out, fell_any = [], False
    for k in range(job.get("ticks", int(EPISODE_S / DT))):
        pushed = k >= 1 and apply_push(duck, ep, k)
        rec = duck.tick(pol, cmd[0:3], cmd[3:7], cmd[7:13])
        mujoco.mj_forward(duck.model, duck.data)
        fell = is_fall(duck)
        out.append(tr.state(duck, k + 1, rec["action"], rec["prev_action"], cmd, fell, fell, pushed))
        if fell:
            fell_any = True
            if job.get("stop_on_fall", True):
                break
    return dict(S=np.stack(out), fell=fell_any)


def traj_of(res: dict) -> dict:
    t = unpack(res["S"])
    t["fell_any"] = res["fell"]
    return t


def run_many(jobs: list[dict], procs: int = 8, pool=None, timeout: float | None = None) -> list[dict]:
    """The episodes in a fresh spawn pool of `procs` processes, or in `pool` (kept open; `timeout` s at most)."""
    if pool is not None:
        return pool.map_async(episode, jobs).get(timeout=timeout)
    with mp.get_context("spawn").Pool(procs) as pool:
        return pool.map(episode, jobs)


def probe_seeds(policy: str, onnx, cmd, hidden: bool = False, procs: int = 8) -> list[dict]:
    eps = [hidden_episode(i) for i in range(N_HIDDEN)] if hidden else [public_episode(i) for i in range(N_PUBLIC)]
    return [traj_of(r) for r in run_many([dict(policy=policy, onnx=onnx, cmd=list(cmd), ep=e) for e in eps], procs)]


def score(trajs: list[dict]) -> dict:
    """All three probes on a set of seed trajectories."""
    from probes import PROBES
    from probes.common import SEED_PASS_FRAC
    res = {}
    for name, p in PROBES.items():
        per = [p.evaluate(t) for t in trajs]
        keys = [k for k in per[0] if k not in ("passed", "no_fall")]
        res[name] = dict(pass_seeds=sum(r["passed"] for r in per), n=len(per),
                         passed=sum(r["passed"] for r in per) >= SEED_PASS_FRAC * len(per),
                         mean={k: round(float(np.mean([r[k] for r in per])), 4) for k in keys},
                         falls=sum(not r["no_fall"] for r in per), seeds=per)
    return res


# ---- replay check: exported ONNX on the replay path vs the training env (numba BAM + torch actor) --------------------
def _train_env_episode(job):
    import torch
    from .learn import load_plain
    torch.set_num_threads(1)
    actor = load_plain(Path(job["ckpt"]))
    env = DuckEnv(job["policy"], job["cmd"], job["cmd"], EPISODE_S, {}, 0)
    obs, _ = env.reset(ep=job["ep"], cmd=job["cmd"])
    out, fell = [], False
    for _ in range(int(EPISODE_S / DT)):
        with torch.no_grad():
            a = actor(torch.from_numpy(obs[None]))[0].numpy()
        obs, _, st, term, _ = env.step(a)
        out.append(st)
        if st[-3] > 0:
            fell = True
            break
    return dict(S=np.stack(out), fell=fell)


def replay_check(move_dir: Path, round_dir: Path) -> dict:
    from . import spec
    move = spec.load(move_dir)
    eps = [public_episode(i) for i in range(N_PUBLIC)]
    cmd = move.eval_cmd.tolist()
    rep = run_many([dict(policy=move.warm_start, onnx=str(round_dir / "policy.onnx"), cmd=cmd, ep=e) for e in eps])
    with mp.get_context("spawn").Pool(8) as pool:
        trn = pool.map(_train_env_episode, [dict(policy=move.warm_start, ckpt=str(round_dir / "ckpt.pt"), cmd=cmd, ep=e)
                                            for e in eps])
    from .duck import LAYOUT
    per = []
    for a, b in zip(rep, trn):
        n = min(len(a["S"]), len(b["S"]), 50)
        o, w = LAYOUT["joint_pos"]
        per.append(dict(max_joint_diff_first_1s=float(np.abs(a["S"][:n, o:o + w] - b["S"][:n, o:o + w]).max()),
                        vx_replay=float(unpack(a["S"])["vx"][50:].mean()), vx_train=float(unpack(b["S"])["vx"][50:].mean()),
                        fell_replay=a["fell"], fell_train=b["fell"]))
    dv = [abs(p["vx_replay"] - p["vx_train"]) for p in per]
    return dict(onnx=str(round_dir / "policy.onnx"), seeds=per, max_abs_vx_diff=round(max(dv), 4),
                mean_vx_replay=round(float(np.mean([p["vx_replay"] for p in per])), 4),
                mean_vx_train=round(float(np.mean([p["vx_train"] for p in per])), 4),
                falls_replay=sum(p["fell_replay"] for p in per), falls_train=sum(p["fell_train"] for p in per),
                max_joint_diff_first_1s=max(p["max_joint_diff_first_1s"] for p in per),
                note="numba BAM == duckref BAM bit for bit (lab/bench.py); remaining differences are torch vs "
                     "onnxruntime float rounding, amplified by chaotic contact dynamics after ~1-4 s")
