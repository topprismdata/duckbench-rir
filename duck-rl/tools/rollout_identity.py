#!/usr/bin/env python
"""The fast rollout's regression test (kit 1.8.0, duck_rl/rollout.py): ROLLOUT "fast" must train exactly as "stock".

  ticks     (light, seconds per Challenge) the fused substep loop vs the stock one, per tick: pairs of envs with the same
            seed (ChallengeEnv, or DuelEnv for Sumo, with fused=False / True) fed the same actions (the stock alpha_walking
            MLP on the stock env's obs + N(0, 0.1) noise, as the PPO sampler at INIT_STD 0.1), auto-reset as the
            workers do; every tick compares qpos, qvel, ctrl, the BAM state (ring, head, previous torque, the model's
            frictionloss), every output of step() (obs, critic extras, the state row, term, time-out) and t_out, bit
            for bit (NaN == NaN), and every reset's obs. Fails if the fused kernel never ran (not available here, or
            the layout check never passed).
  rollouts  (heavy: two short trainings; run it through `tools/heavy --lane rl`) train.py --rollout stock, then
            --rollout fast, same config and --seed, no TS eval: the rollout storage (obs, critic obs, actions, rewards,
            dones, values, log-probs, distribution params) is hashed per PPO rollout just before compute_returns; every
            rollout both runs reached must hash the same (at least --min-rollouts of them, past the critic warm-up, so
            updated weights are covered).

    PY=~/.duckbench/duck-rl/.venv/bin/python   # (a player's kit: uv run python)
    $PY tools/rollout_identity.py ticks                                  # sprint hurdles turntable wheel saucer sumo crossy-duck red-light
    $PY tools/rollout_identity.py ticks --challenges saucer --ticks 2000
    tools/heavy --lane rl -- $PY spikes/duck-rl/kit/tools/rollout_identity.py rollouts --config configs/sprint/stock.py

Prints one JSON line per Challenge (or run pair) and exits 1 on any mismatch. Light by design: ticks is ~5-20 s per Challenge
(mostly the numba compile and the TS scene export on first use); rollouts ~1.5 min per config at --minutes 0.3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
CHALLENGES = ("sprint", "hurdles", "turntable", "wheel", "saucer", "sumo", "crossy-duck", "red-light")


def _setup() -> None:
    os.chdir(KIT)
    if str(KIT) not in sys.path:
        sys.path.insert(0, str(KIT))


def same(a, b) -> bool:
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


# ---- ticks ------------------------------------------------------------------------------------------------------------
def _pair(challenge: str, seed: int):
    """(stock env, fused env, physics views) for a Challenge: a walk node for a solo Challenge (the substep loop does not depend
    on the node), configs/sumo/baseline.py's duel for a duel Challenge."""
    import duck_rl
    info = duck_rl.load_info(challenge)
    if "duel" in info:
        import train
        from duck_rl.duel_env import DuelEnv
        cfg_path = KIT / "configs" / challenge / "baseline.py"
        cfg = train.duel_config(train.load_config(cfg_path), challenge)
        s, f = DuelEnv(cfg, seed), DuelEnv(cfg, seed, fused=True)
        views = lambda e: (e.world.m, e.world.d, e.world.bams, e.world._fused)  # noqa: E731
        return s, f, views
    from duck_rl.env import ChallengeEnv
    scene, _ = duck_rl.ensure_scene(challenge)
    cfg = dict(node={"skill": "walk", "params": {"vx": 0.15}}, info=info, scene=str(scene))
    cfg_path = KIT / "configs" / challenge / "baseline.py"
    if "runtime" in info and cfg_path.exists():  # a Challenge with a runtime mechanic: its baseline's env (program mode)
        import train
        C = train.load_config(cfg_path)
        compiled = train.compile_program(challenge, train.resolve(C.PROGRAM), duck_rl.HOME / "tmp" / f"rollout-id-{challenge}.json")
        cfg = train.env_config(C, challenge, compiled)
    s, f = ChallengeEnv(cfg, seed), ChallengeEnv(cfg, seed, fused=True)
    views = lambda e: (e.duck.model, e.duck.data, [e.duck.bam], e._fused)  # noqa: E731
    return s, f, views


def ticks_one(challenge: str, n_envs: int, n_ticks: int) -> dict:
    import numpy as np
    import lab
    from duckref.mlp import DuckMlp
    mlp = DuckMlp(lab.PLANT / "policies" / "alpha_walking.mlp.json")
    pairs = [_pair(challenge, 100_000 + i) for i in range(n_envs)]
    bad = {k: 0 for k in ("reset", "qpos", "qvel", "ctrl", "bam", "obs", "priv", "S", "term", "tout", "t_out")}
    first_bad, resets, touches = None, 0, 0
    obs = []
    for s, f, _ in pairs:
        os_, of_ = s.reset(), f.reset()
        bad["reset"] += not (same(os_[0], of_[0]) and same(os_[1], of_[1]))
        obs.append(os_[0])
    obs = np.stack(obs)
    rng = np.random.default_rng(7)
    for t in range(n_ticks):
        act = (mlp(obs) + rng.normal(0.0, 0.1, size=(n_envs, 14))).astype(np.float32)
        for i, (s, f, views) in enumerate(pairs):
            out0 = getattr(s, "t_out", None)
            rs, rf = s.step(act[i]), f.step(act[i])
            touches += out0 is None and getattr(s, "t_out", None) is not None
            ms, ds, bs, _ = views(s)
            mf, df, bf, _ = views(f)
            chk = dict(qpos=same(ds.qpos, df.qpos), qvel=same(ds.qvel, df.qvel), ctrl=same(ds.ctrl, df.ctrl),
                       bam=all(same(a.ring, b.ring) and a.head == b.head and same(a.prev_torque, b.prev_torque)
                               for a, b in zip(bs, bf)) and same(ms.dof_frictionloss, mf.dof_frictionloss),
                       obs=same(rs[0], rf[0]), priv=same(rs[1], rf[1]), S=same(rs[2], rf[2]), term=rs[3] == rf[3],
                       tout=rs[4] == rf[4], t_out=getattr(s, "t_out", None) == getattr(f, "t_out", None))
            for k, v in chk.items():
                if not v:
                    bad[k] += 1
                    first_bad = first_bad or dict(tick=t, env=i, field=k)
            o = rs[0]
            if rs[3] or rs[4]:
                resets += 1
                os_, of_ = s.reset(), f.reset()
                bad["reset"] += not (same(os_[0], of_[0]) and same(os_[1], of_[1]))
                o = os_[0]
            obs[i] = o
    kernels = [views(f)[3] for _, f, views in pairs]
    fused_on = all(k is not None and k.ok for k in kernels)
    parity = all(v == 0 for v in bad.values())
    return dict(challenge=challenge, envs=n_envs, ticks=n_ticks, resets=resets, floor_touches=touches, fused_ran=fused_on,
                parity=parity, ok=parity and fused_on, mismatches={k: v for k, v in bad.items() if v}, first_bad=first_bad)


def cmd_ticks(a) -> int:
    _setup()
    import duck_rl  # noqa: F401  (paths, the lab)
    from duck_rl.fused import available
    ok, why = available()
    if not ok:
        print(json.dumps(dict(ok=False, reason=f"the fused kernel is not available here: {why}")), flush=True)
        return 1
    fails = 0
    for challenge in a.challenges.split(","):
        r = ticks_one(challenge, a.envs, a.ticks)
        print(json.dumps(r), flush=True)
        fails += not r["ok"]
    return 1 if fails else 0


# ---- rollouts ---------------------------------------------------------------------------------------------------------
def cmd_train(a) -> int:
    """(internal) train.py's main() with the rollout storage hashed per PPO rollout into a.hash_out."""
    _setup()
    sys.argv = [str(KIT / "train.py"), *a.rest]
    import train  # noqa: F401
    from rsl_rl.algorithms import PPO
    fh = open(a.hash_out, "w")
    orig = PPO.compute_returns
    n = [0]

    def h(x) -> str:
        return hashlib.sha256(x.numpy().tobytes()).hexdigest()[:16]

    def compute_returns(self, obs):
        st = self.storage
        rec = {"it": n[0], **{f"obs.{k}": h(st.observations[k]) for k in sorted(st.observations.keys())}}
        rec.update({k: h(getattr(st, k)) for k in ("actions", "rewards", "dones", "values", "actions_log_prob")})
        rec.update({f"dist{i}": h(p) for i, p in enumerate(st.distribution_params or ())})
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        n[0] += 1
        return orig(self, obs)
    PPO.compute_returns = compute_returns  # (the test's own hook; the shipped path has none)
    return train.main() or 0


def cmd_rollouts(a) -> int:
    _setup()
    import duck_rl
    cfg = Path(a.config)
    tag = f"rollout-id-{cfg.parent.name}-{cfg.stem}"
    out_dir = duck_rl.HOME / "rollout-identity"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for mode in ("stock", "fast"):
        label = f"{tag}-{mode}"
        hf = out_dir / f"{label}.jsonl"
        log = out_dir / f"{label}.log"
        cmd = [sys.executable, str(Path(__file__).resolve()), "_train", str(hf), "--", "--config", str(cfg),
               "--label", label, "--rollout", mode, "--seed", str(a.seed), "--minutes", str(a.minutes),
               "--no-ts-eval", "--no-save-runs"]
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=KIT).returncode
        for d in (duck_rl.RUNS / label, KIT / "submissions" / label):
            shutil.rmtree(d, ignore_errors=True)
        rs = [json.loads(x) for x in hf.read_text().splitlines()] if hf.exists() else []
        runs[mode] = rs
        if rc != 0:
            print(json.dumps(dict(config=str(cfg), ok=False, reason=f"{mode} run exited {rc} (log {log})")), flush=True)
            return 1
    ref, fast = runs["stock"], runs["fast"]
    n = min(len(ref), len(fast))
    diff = next((i for i in range(n) if ref[i] != fast[i]), None)
    fields = sorted(k for k in ref[diff] if ref[diff][k] != fast[diff].get(k)) if diff is not None else []
    ok = diff is None and n >= a.min_rollouts
    print(json.dumps(dict(config=str(cfg), ok=ok, rollouts_stock=len(ref), rollouts_fast=len(fast), compared=n,
                          identical=diff if diff is not None else n, first_diff_fields=fields,
                          min_rollouts=a.min_rollouts, logs=str(out_dir))), flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("ticks", help="fused vs stock substep loop, per tick (light)")
    t.add_argument("--challenges", default=",".join(CHALLENGES))
    t.add_argument("--dares", dest="challenges", default=argparse.SUPPRESS, help=argparse.SUPPRESS)  # rename-challenge:keep: the pre-rename --dares, a silent alias
    t.add_argument("--envs", type=int, default=2)
    t.add_argument("--ticks", type=int, default=500)
    r = sub.add_parser("rollouts", help="rollout-storage hashes, stock vs fast (heavy: tools/heavy --lane rl)")
    r.add_argument("--config", required=True, help="a kit config (relative to the kit, or absolute)")
    r.add_argument("--minutes", type=float, default=0.3)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--min-rollouts", type=int, default=15)
    tr = sub.add_parser("_train")
    tr.add_argument("hash_out")
    tr.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.cmd == "_train":
        a.rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
        return cmd_train(a)
    return cmd_ticks(a) if a.cmd == "ticks" else cmd_rollouts(a)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # as train.py's own __main__: skip the C++ static destructors (the exit-134 teardown abort)
