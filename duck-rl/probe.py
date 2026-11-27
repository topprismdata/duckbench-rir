#!/usr/bin/env python
"""Roll a policy (stock alpha_walking or a duck-mlp/1 file) through a config's training episodes in the Python env
(its bank starts or spawn draws, its schedule, deterministic actions) and print the metric distribution. A code check
of the env and reward before a run, not training and not the score (the TS eval is). From opus's kit v1 changes.
A duel Challenge (Sumo): training matches against the config's OPPONENT (duck_rl/duel_env.py), the policy on its trained
nodes through the sim's runner (SimdMlp: the stock policy = the duckscript as it is), and the wins, draws, losses,
unbound ends, the judge's end reasons and the mean return per term.
In a player's kit it charges the owner's budget (duck_rl/budget.py).

    python probe.py --config configs/turntable/baseline.py [--mlp submissions/x/policy.mlp.json] [--n 32]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

T0 = time.time()  # charged from here (the imports count)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import duck_rl  # noqa: E402
from duck_rl import budget  # noqa: E402

budget.begin("probe", T0)  # the budget record goes up before the heavy imports

import numpy as np  # noqa: E402
from duck_rl.env import load_bank, make_env, make_unpack  # noqa: E402
from duck_rl.rank import rank_result  # noqa: E402
from duck_rl.simd_mlp import SimdMlp  # noqa: E402  (kit 1.16.0: the sim's float32 runner; was duckref DuckMlp)
from train import compile_program, config_challenge, env_config, load_config, resolve, reward_fn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--mlp", default=None)
ap.add_argument("--n", type=int, default=32)
a = ap.parse_args()
cfg_path = Path(a.config).resolve()
C = load_config(cfg_path)
challenge = config_challenge(C, cfg_path)
compiled = compile_program(challenge, resolve(C.PROGRAM), duck_rl.HOME / "tmp" / "probe-compiled.json")
cfg = env_config(C, challenge, compiled)
info = cfg["info"]
unpack = make_unpack(info)
terms = reward_fn(C, dict(getattr(C, "REWARD_PARAMS", {})))
pol = SimdMlp(Path(a.mlp) if a.mlp else duck_rl.PLANT / "policies" / "alpha_walking.mlp.json")
bank = load_bank(cfg["bank"]) if cfg.get("start") == "handoff" and cfg.get("bank") else []
rows = []
if "duel" in info:  # a duel Challenge: matches, then its own summary
    from duck_rl.duel import load_runner  # noqa: E402
    runner = load_runner(Path(a.mlp).resolve() if a.mlp else duck_rl.PLANT / "policies" / "alpha_walking.mlp.json")
    act = np.empty(14, np.float32)
    with budget.charged("probe", cfg_path.name, start=T0):
        env = make_env(cfg, 7)
        for i in range(a.n):
            obs, _ = env.reset()
            t0, S = env.k * 0.02, []
            while True:
                runner.run(obs, act)
                obs, _, st, term, to = env.step(act)
                S.append(st)
                if term or to:
                    break
            ret = {k: float(np.sum(v) * 0.02) for k, v in terms(unpack(np.stack(S))).items()}
            rows.append(dict(episode=i, opponent=env.refs[env.cur], swapped=env.swapped, t0=t0, t_end=env.k * 0.02,
                             end=env.end_reason, reason=env.row.reason, points=env.score, ret=ret))
    print(json.dumps(rows[:4], indent=0))
    n_of = {k: sum(r["end"] == k for r in rows) for k in ("won", "draw", "lost", "invalid", "unbound")}
    dec = [r["points"] for r in rows if r["end"] in ("won", "draw", "lost", "invalid")]
    why = {}
    for r in rows:
        why[r["reason"] or "-"] = why.get(r["reason"] or "-", 0) + 1
    print(f"{challenge} n {len(rows)} vs {sorted(set(r['opponent'] for r in rows))}: {n_of['won']} W {n_of['draw']} D "
          f"{n_of['lost']} L, {n_of['invalid']} invalid, {n_of['unbound']} unbound (left the trained nodes first); mean "
          f"points of the decided {np.mean(dec) if dec else float('nan'):.3f}; mean length {np.mean([r['t_end'] for r in rows]):.2f} s; "
          f"judge's end reasons {why}; return per term (mean): "
          + ", ".join(f"{k} {np.mean([r['ret'][k] for r in rows]):.2f}" for k in rows[0]["ret"])
          + f"; total {np.mean([sum(r['ret'].values()) for r in rows]):.2f}")
    sys.exit(0)
with budget.charged("probe", cfg_path.name, start=T0):
    env = make_env(cfg, 7)
    for i in range(min(a.n, len(bank)) if bank else a.n):
        obs, _ = env.reset(bank_state=bank[i]) if bank else env.reset()
        t0 = env.k * 0.02
        S = []
        while True:
            act = np.asarray(pol(obs.astype(np.float32)), np.float32).reshape(14)
            obs, _, st, term, to = env.step(act)
            S.append(st)
            if term or to:
                break
        D = unpack(np.stack(S))
        ret = {k: float(np.sum(v) * 0.02) for k, v in terms(D).items()}
        rows.append(dict(seed=bank[i]["seed"] if bank else i, t0=t0, t_end=float(D["t"][-1]), metric=float(env.score),
                         lost=bool(D["lost"][-1]), success=bool(D["success"][-1]), fell=bool(D["fell"][-1]), ret=ret,
                         **({"finish_t": float(D["finish_t"][-1])} if "finish_t" in D else {})))  # a race: its result
sc = np.array([r["metric"] for r in rows])
print(json.dumps(rows[:4], indent=0))
print(f"{challenge} n {len(rows)}: metric at end median {np.nanmedian(sc):.3f} mean {np.nanmean(sc):.3f} [{np.nanmin(sc):.3f}, "
      f"{np.nanmax(sc):.3f}], success {sum(r['success'] for r in rows)}, lost {sum(r['lost'] for r in rows)}, "
      f"fell {sum(r['fell'] for r in rows)}; return per term (median): "
      + ", ".join(f"{k} {np.median([r['ret'][k] for r in rows]):.2f}" for k in rows[0]["ret"]))
if "finish_t" in rows[0]:  # a race (max_distance, torso_reach): the seed's result = the finish time, 30 s if it never finished (D-19)
    ft = np.array([r["finish_t"] for r in rows])
    bh = rank_result(ft.tolist(), lower_is_better=True)  # the ranking number (kit 1.14.0): the mean of the best half
    print(f"race: finish_t best {(len(ft) + 1) // 2} of {len(ft)} {bh:.2f} s, mean {ft.mean():.2f} s, worst {ft.max():.2f} s, "
          f"finished {int(np.sum(ft < 30.0))}/{len(ft)} (an episode cut before the line or by a fall counts 30 s)")
elif np.all(np.isfinite(sc)):  # a score Challenge: the ranking number is the best-half mean of the metric
    print(f"score: best {(len(sc) + 1) // 2} of {len(sc)} {rank_result(sc.tolist(), lower_is_better=False):.3f} (the ranking number)")
