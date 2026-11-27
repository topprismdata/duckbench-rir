#!/usr/bin/env python
"""Port check for any Challenge: ChallengeEnv (Python, numba BAM) vs the TS sim on the same states, per seed:
  (1) the spawn reset with that seed's TS draws vs TS row 0 (exact up to the trajectory's float32 rounding), and the
      metric column;
  (2) from the TS handoff state (a bank at --node, recorded by ts/eval.ts --bank), --ticks ticks of the stock
      alpha_walking policy vs the TS run of the same seed (full program, the same node over the same ticks): qpos, the
      Challenge metric (the trajectory's metric column vs the env's `metric`).
After a few seconds of walking the runs diverge chaotically (float rounding: native vs wasm MuJoCo); since kit 1.16.0 (the
wasm-compiled world, SimdMlp as the policy) 1 tick agrees to the trajectory's float32 rounding.

The raw host (--raw BLOCK, a command block for --node as a strategy writes it: '{"vx": 0.65, "head_pitch": 0.2}' or 13
numbers; slot names): TS runs --mlp on the node bound raw (eval.ts --move, for the bank and the trajectories alike),
Python runs the same weights with ChallengeEnv host "raw" (controller.RawController, head EMA in the obs builder). Also
printed: the error had the Python side used the walk-stand controller for that node (the check's sensitivity: what a
wrong mirror looks like). --done RULE ('{"after_s": 1}', either host) adds the move's done rule on both sides and
compares when the node raised done (TS: the episode's done event; Python: the first tick of the move_done column; run
--ticks past it).

A race (metric torso_reach: sprint, hurdles, steps; or max_distance: crossy-duck, red-light) also gets (3) the mirror
replay: duck_rl/metrics.py TorsoReach / MaxDistance fed the
TS trajectory's rows (qpos, float32) over the whole episode, from row 0 and from the handoff row restored from the
bank's metric_tail (ts/metric_state.ts), vs the trajectory's metric column per row and the TS time to clear vs the
mirror's finish_t (the formula and the tail, free of the physics' chaotic divergence). --episode adds (4): the Python
env itself from the spawn (stock policy or --mlp on the start node, no fall rule) vs TS, row by row, while TS stays on
that node (a fall sends the program to get_up, which the env cannot run): the metric column, finished / finish_t.

    python tools/parity.py --challenge sprint --seeds 0-3 --episode       # the stock clever, its `walk` node
    python tools/parity.py --challenge hurdles --program programs/hurdles.json --node run --seeds 0-3 --episode
    python tools/parity.py --challenge turntable [--program <stock clever> --node backpedal --seeds 0-3 --ticks 1]
    python tools/parity.py --challenge sprint --program P --node run --mlp x.mlp.json --raw '{"vx": 0.65, "head_pitch": 0.2}' \
        --done '{"after_s": 1}' --ticks 1,5,25,60
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import duck_rl  # noqa: E402
from duck_rl import bridge  # noqa: E402
from duck_rl.controller import command_vector  # noqa: E402
from duck_rl.env import DT, ChallengeEnv, columns  # noqa: E402
from duck_rl.metrics import make_metric  # noqa: E402
from duck_rl.simd_mlp import SimdMlp  # noqa: E402  (kit 1.16.0: the sim's float32 runner, bit for bit; DuckMlp drifts ~1e-6)
from train import compile_program, resolve  # noqa: E402

DEFAULT_NODE = {"turntable": "backpedal", "wheel": "run", "sprint": "walk", "hurdles": "go"}  # the stock clever programs' handoff nodes

ap = argparse.ArgumentParser()
_challenge = ap.add_mutually_exclusive_group(required=True)
_challenge.add_argument("--challenge")
_challenge.add_argument("--dare", dest="challenge", help=argparse.SUPPRESS)  # rename-challenge:keep: the pre-rename --dare, a silent alias
ap.add_argument("--program", default=None, help="default packages/challenges/baselines/<challenge>/clever.json (the monorepo)")
ap.add_argument("--node", default=None, help="the walk/stand node to hand off at (default: the challenge's, else the start node)")
ap.add_argument("--seeds", default="0-3")
ap.add_argument("--ticks", default="1", help="tick counts after the handoff to compare at (comma-separated)")
ap.add_argument("--mlp", default=None, help="a duck-mlp/1 x.mlp.json to run on the node (default the stock alpha_walking)")
ap.add_argument("--raw", default=None, help="the node's raw command block (JSON; slot names or 13 numbers): host raw")
ap.add_argument("--done", default=None, help="the move's done rule (JSON {after_s?, upright_s?, rolled_then_upright_s?})")
ap.add_argument("--episode", action="store_true", help="also run the env from the spawn while TS stays on the start node")
a = ap.parse_args()
ticks = sorted({int(x) for x in a.ticks.split(",")})
mlp = Path(a.mlp).resolve() if a.mlp else duck_rl.PLANT / "policies" / "alpha_walking.mlp.json"
raw = json.loads(a.raw) if a.raw else None
done = json.loads(a.done) if a.done else None
if a.program is None and duck_rl.REPO is None:
    ap.error("--program: no repo here for the stock clever duckscript")
program = resolve(a.program or f"packages/challenges/baselines/{a.challenge}/clever.json")
scene, _ = duck_rl.ensure_scene(a.challenge)
info = duck_rl.load_info(a.challenge)
nq = info["nq"]
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    compiled = compile_program(a.challenge, program, td / "compiled.json")
    node_id = a.node or DEFAULT_NODE.get(a.challenge) or compiled["start"]
    bind = []  # the move on the node, TS side (none: the stock policy runs there)
    if a.mlp or raw is not None or done is not None:
        mv = {"nodes": [node_id], "move": str(mlp), **({"host": "raw", "command": {node_id: raw}} if raw is not None else {}),
              **({"done": done} if done is not None else {})}
        bind = ["--move", json.dumps(mv)]
    r = bridge.run_ts("eval", ["--challenge", a.challenge, "--program", program, *bind, "--seeds", a.seeds, "--bank", td / "bank.json",
                               "--bank-nodes", node_id, "--quiet"], capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr[-3000:])
    bank = {s["seed"]: s for s in json.loads((td / "bank.json").read_text())["states"]}
    seeds = sorted(bank)
    r = bridge.run_ts("eval", ["--challenge", a.challenge, "--program", program, *bind, "--seeds", a.seeds, "--traj-dir", td / "traj",
                               "--traj", ",".join(map(str, seeds)), "--out", td / "sum.json", "--quiet"],
                      capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr[-3000:])
    per_seed = {row["seed"]: row for row in json.loads((td / "sum.json").read_text())["per_seed"]}
    ts_done = {s: [t for t, n in row.get("done", []) if n == node_id] for s, row in per_seed.items()}
    traj = {}
    for s in seeds:
        blob = (td / "traj" / f"s{s:04d}.traj").read_bytes()
        tnq, n = struct.unpack_from("<II", blob, 8)
        traj[s] = np.frombuffer(blob, dtype="<f4", offset=16).reshape(n, tnq + 3)
node = dict(compiled["nodes"][node_id], id=node_id)
pol = SimdMlp(mlp)
base = dict(node=node, info=info, scene=str(scene), fall="none", end_on_out=False, end_s=info["duration_s"],
            **({"done": done} if done is not None else {}))
if json.loads(Path(program).read_text()).get("format") == "duck-program/2":  # kit 1.15.0: the node's feed / head holds
    from duck_rl.duel import compile_program as compile_full
    base["program"] = compile_full(a.challenge, program)
env = ChallengeEnv(dict(base, **({"host": "raw", "raw_command": {node_id: command_vector(raw).tolist()}} if raw is not None else {})), 0)
# the sensitivity check: the same run had Python fed the node's walk-stand command block instead of the raw one
wrong = ChallengeEnv(base, 0) if raw is not None else None
EX = columns(info)
race = info["metric"]["kind"] in ("max_distance", "torso_reach")


def cos_tilt_of(row: np.ndarray) -> float:
    """The trunk's xmat[8] from a trajectory row's free-joint quaternion (normalised, as mj_kinematics does)."""
    w, x, y, z = (float(v) for v in row[3:7])
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return w * w - x * x - y * y + z * z


def replay(f: np.ndarray, row0: int = 0, tail: dict | None = None) -> dict:
    """The metric mirror alone over TS rows row0.. (restored from `tail` at row0 > 0): max |metric - TS metric|, the
    first row it differs by > 1e-5, its finish_t (None if never) and final value."""
    m = make_metric(info, DT)
    if row0 > 0:
        m.restore(tail, row0)
    dmax, first_bad = 0.0, None
    for r in range(row0, f.shape[0]):
        m.push(f[r, :nq].astype(np.float64), None, cos_tilt=cos_tilt_of(f[r]))
        dm = abs(m.value - float(f[r, nq]))
        dmax = max(dmax, dm)
        if first_bad is None and dm > 1e-5:
            first_bad = r
    return dict(d_metric=dmax, first_bad_row=first_bad, finish_t=m.finish_t, value=m.value)


def episode(b: dict, f: np.ndarray) -> dict:
    """The env from the spawn (this seed's draws) while TS stays on the start node: metric / finished per row."""
    obs, _ = env.reset(ep=b["draws"])
    n0 = f[0, nq + 1]
    last = next((r for r in range(1, f.shape[0]) if f[r, nq + 1] != n0), f.shape[0]) - 1  # TS's last row on the node
    dmax, fin_py, dq = 0.0, None, 0.0
    for k in range(1, last + 1):
        obs, _, st, term, to = env.step(pol(obs).astype(np.float32))
        dmax = max(dmax, abs(float(st[EX["metric"]]) - float(f[k, nq])))
        if fin_py is None and st[EX["finished"]] > 0:
            fin_py = float(st[EX["finish_t"]])
        dq = float(np.abs(env.duck.data.qpos[:2] - f[k, :2]).max())
        if term or to:
            break
    return dict(rows=k, last_t=round(k * DT, 2), d_metric=dmax, finish_py=fin_py, d_xy_end=dq,
                metric_py=float(st[EX["metric"]]), metric_ts=float(f[k, nq]))


def run(e: ChallengeEnv, b: dict, f: np.ndarray) -> tuple[dict, int | None]:
    """Step `e` from the bank state; per tick count in `ticks`: qpos vs the TS row; the first move_done tick."""
    obs, _ = e.reset(bank_state=b)
    k0, out, done_k = int(b["tick"]), {}, None
    for j in range(1, ticks[-1] + 1):
        obs, _, st, term, to = e.step(pol(obs).astype(np.float32))
        if done_k is None and st[EX["move_done"]] > 0:
            done_k = k0 + j
        if j in ticks:
            k1 = k0 + j
            diff = np.abs(e.duck.data.qpos - f[k1, :nq].astype(np.float64))
            out[j] = dict(dq=float(diff.max()), dxy=1000 * float(np.hypot(*(e.duck.data.qpos[:2] - f[k1, :2]))),
                          djoint=float(diff[e.duck.qadr].max()), metric=(float(st[EX["metric"]]), float(f[k1, nq])))
    return out, done_k


rows = []
for s in seeds:
    b, f = bank[s], traj[s]
    env.reset(ep=b["draws"])
    d0 = float(np.abs(env.duck.data.qpos - f[0, :nq].astype(np.float64)).max())
    m0 = (env._metric_now(), float(f[0, nq]))
    after, done_k = run(env, b, f)
    bad = run(wrong, b, f)[0] if wrong is not None else None
    tsd = ts_done.get(s, [])
    t_after = [t for t in tsd if t >= b["t"] - 1e-9]
    rows.append(dict(seed=s, handoff_t=b["t"], d_qpos_row0=d0, metric_row0=m0, after=after, wrong=bad,
                     done_py=None if done_k is None else round(done_k * DT, 4), done_ts=t_after[0] if t_after else None))
    print(f"seed {s}: reset vs TS row 0: max |dqpos| {d0:.2e}, metric {m0[0]:.4f} vs {m0[1]:.4f}; handoff at {b['t']:.2f} s ({node_id}"
          + (f", raw {raw}" if raw is not None else "") + ")")
    for j, r in after.items():
        print(f"  + {j:4d} tick(s): max |dqpos| {r['dq']:.2e} (trunk xy {r['dxy']:.3f} mm, joints {r['djoint']:.1e} rad), metric {r['metric'][0]:.4f} vs {r['metric'][1]:.4f}"
              + (f"; walk-stand mirror instead: max |dqpos| {bad[j]['dq']:.2e} (trunk xy {bad[j]['dxy']:.3f} mm)" if bad else ""))
    if done is not None:
        print(f"  done: Python move_done at {rows[-1]['done_py']} s, TS done event at {rows[-1]['done_ts']} s")
    if race:
        ttc = per_seed[s].get("finish_t", per_seed[s].get("time_to_clear"))  # eval.ts: finish_t (kit 1.4), was time_to_clear
        k0 = int(b["tick"])
        rp0, rpb = replay(f), replay(f, k0, b.get("metric_tail"))
        rows[-1].update(ts_ttc=ttc, ts_score=per_seed[s]["score"], replay0=rp0, replay_bank=rpb)
        print(f"  mirror replay of the TS rows: from row 0 max |dmetric| {rp0['d_metric']:.1e} (final {rp0['value']:.4f} vs TS "
              f"{per_seed[s]['score']:.4f}), finish_t {rp0['finish_t']} vs TS time to clear {ttc}; from the bank row {k0} "
              f"(tail {'none' if not b.get('metric_tail') else 'restored'}): max |dmetric| {rpb['d_metric']:.1e}, finish_t {rpb['finish_t']}")
    if a.episode:
        assert node_id == compiled["start"], f"--episode: --node must be the start node ({compiled['start']})"
        ep = episode(b, f)
        rows[-1]["episode"] = ep
        print(f"  env episode from the spawn, {ep['rows']} rows (to {ep['last_t']} s, TS on {node_id}): max |dmetric| {ep['d_metric']:.4f}, "
              f"final metric {ep['metric_py']:.4f} vs TS {ep['metric_ts']:.4f}, trunk xy off {1000 * ep['d_xy_end']:.1f} mm; "
              + (f"finish_t {ep['finish_py']} vs TS {rows[-1]['ts_ttc']}" if race else ""))
summary = dict(challenge=a.challenge, node=node_id, mlp=str(mlp), raw=raw, done=done, ticks=ticks,
               max_d_qpos_row0=max(r["d_qpos_row0"] for r in rows),
               max_d_qpos_after={j: max(r["after"][j]["dq"] for r in rows) for j in ticks},
               max_d_xy_mm_after={j: max(r["after"][j]["dxy"] for r in rows) for j in ticks},
               max_d_metric_after={j: max(abs(r["after"][j]["metric"][0] - r["after"][j]["metric"][1]) for r in rows) for j in ticks})
if wrong is not None:
    summary["walk_stand_mirror_max_d_qpos_after"] = {j: max(r["wrong"][j]["dq"] for r in rows) for j in ticks}
if race:
    def _fin(x, y):  # finish times: both None, or equal to the row
        return 0.0 if x is None and y is None else (float("inf") if x is None or y is None else abs(x - y))
    summary["replay_max_d_metric_row0"] = max(r["replay0"]["d_metric"] for r in rows)
    summary["replay_max_d_metric_bank"] = max(r["replay_bank"]["d_metric"] for r in rows)
    summary["replay_max_d_finish_t"] = max(_fin(r["replay0"]["finish_t"], r["ts_ttc"]) for r in rows)
    summary["replay_bank_max_d_finish_t"] = max(_fin(r["replay_bank"]["finish_t"], r["ts_ttc"]) for r in rows)
if a.episode:
    summary["episode_max_d_metric"] = max(r["episode"]["d_metric"] for r in rows)
    summary["episode_rows"] = [r["episode"]["rows"] for r in rows]
    if race:
        summary["episode_finish"] = [(r["episode"]["finish_py"], r["ts_ttc"]) for r in rows]
if done is not None:
    summary["done_match"] = all(r["done_py"] is not None and r["done_ts"] is not None and abs(r["done_py"] - r["done_ts"]) < 1e-6 for r in rows)
print(json.dumps(summary))
