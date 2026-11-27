#!/usr/bin/env python
"""Local TS eval on any seeds: the official eval code (the pinned repo's spikes/duck-rl/eval_submission.py +
ts/eval.ts), run on your submission, or on a duckscript with trained policies. It is Mac time: run it under the budget
runner (`budget.py run --label ev -- .venv/bin/python tools/eval_ts.py ...`, or `budget.py eval`, the same thing).
In a player's kit it runs the CLI's `kit eval` (the same ts/eval.ts; no videos: `duckbench simulate` and clips are the
CLI's) and charges the owner's budget (duck_rl/budget.py).

    tools/eval_ts.py --submission submissions/x [--seeds 0-31] [--videos none|median,best|7] [--out home/evals/NAME]
    tools/eval_ts.py --challenge turntable --program packages/challenges/baselines/turntable/clever.json \
        [--policy backpedal=REF]... [--seeds 0-31] ...
    tools/eval_ts.py --submission submissions/x --opponent house:easy --opponent house:medium [--seeds 0-31]   (Sumo)

A submission is any format the eval reads (duck-strategy/1, or the legacy duck-rl-submission/1 / disc-rl-submission/1).
REF is a move ref: a path to x.mlp.json (x.mlp.bin beside it), a move id mv_<12 hex> or an alias <slug>@<n> of the
move library (resolved by the kit's move index, duck_rl/moves.py); the moves are copied into a temporary
duck-strategy/1 (the legacy duck-rl-submission/1 for a spike Challenge, version 0) with each move pinned to its id.

Writes OUT/summary.json (per-seed scores + diagnostics, the natural results and the rank key, D-19), OUT/result.md and, unless --videos none, the
median/best videos (monorepo only). Prints the one-line result. Cost (quiet Mac): ~0.2 s CPU per seed for the TS
episodes, plus ~20-25 s per video.

A duel Challenge (Sumo) is scored by matches: each seed from both start spots against each --opponent (repeatable:
house:easy, house:medium, stand, or a duckscript / strategy file or a submission dir; default house:medium); the
result is the mean points per round (W/D/L, a BT rating estimate against the house bots in summary.json's `duel`), no
videos. Cost: ~0.4 s CPU per round, 2 rounds per seed and opponent.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

T0 = time.time()  # charged from here (the imports count)
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import duck_rl  # noqa: E402
from duck_rl import bridge, budget, moves  # noqa: E402
from duck_rl.rank import rank_result  # noqa: E402

if __name__ == "__main__":
    budget.begin("eval", T0)  # the budget record goes up before the heavy imports (train: torch, mujoco)

from train import resolve  # noqa: E402


def duel_line(s: dict) -> str:
    """A duel's result: '43/64 pts (mean 0.672), 40 W · 6 D · 18 L, worst seed 0.00 vs house:easy; BT estimate 1236'."""
    d, bt = s["duel"], s["duel"].get("bt_estimate")
    tally = lambda t: (f"{t['points']:g}/{t['rounds']} pts (mean {t['points'] / max(1, t['rounds']):.3f}), "  # noqa: E731
                       f"{t['wins']} W · {t['draws']} D · {t['losses']} L" + (f" · {t['invalid']} invalid" if t["invalid"] else ""))
    ops = d["by_opponent"]
    vs = (f" vs {next(iter(ops))}" if len(ops) == 1 else
          " [" + "; ".join(f"vs {k}: {tally(v)}" for k, v in ops.items()) + "]")
    est = f"BT estimate {bt['rating']:.0f} (not the server's rating)" if bt else "BT estimate - (no house opponent)"
    worst = "-" if s["worst_result"] is None else f"{s['worst_result']:.2f}"
    return f"{tally(d)}, worst seed {worst}{vs}; {est}"


def solo_line(s: dict) -> str:
    """A solo eval's result (kit 1.14.0): the ranking number, the mean of the best half of the seeds, then the plain mean
    and the worst seed: 'best 4 of 8 4.10 s (mean 4.21 s, finished 7/8, worst 6.90 s)'."""
    f = lambda v, p: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
    if s.get("crowd"):  # a crowd Challenge keeps the plain mean points
        return f"mean {f(s['mean_result'], 3)} {s['result_unit']} per round, worst {f(s['worst_result'], 3)}"
    lower = s.get("better") == "lower" or s["result_kind"] == "race"
    res = s["result"] if "result" in s else rank_result([r.get("result") for r in s.get("per_seed") or []], lower)
    head = f"best {(s['n'] + 1) // 2} of {s['n']}"
    if s["result_kind"] == "race":
        return f"{head} {f(res, 2)} s (mean {f(s['mean_result'], 2)} s, finished {s['finished']}/{s['n']}, worst {f(s['worst_result'], 2)} s)"
    u = s["result_unit"]
    return f"{head} {f(res, 3)} {u} (mean {f(s['mean_result'], 3)}, worst {f(s['worst_result'], 3)} {u})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission")
    ap.add_argument("--challenge", help="the challenge (required with --program; with --submission, a check)")
    ap.add_argument("--dare", dest="challenge", default=argparse.SUPPRESS, help=argparse.SUPPRESS)  # rename-challenge:keep: the pre-rename --dare, a silent alias
    ap.add_argument("--program")
    ap.add_argument("--policy", action="append", default=[], help="nodes=REF (with --program): a path to x.mlp.json, a move id or an alias")
    ap.add_argument("--seeds", default="0-31")
    ap.add_argument("--videos", default="none")
    ap.add_argument("--out", default=None)
    ap.add_argument("--opponent", action="append", default=[],
                    help="a duel challenge: house:easy | house:medium (the default) | house:hard (Cliff) | stand | a duckscript / strategy file or "
                         "submission dir; repeatable")
    a = ap.parse_args()
    if (a.submission is None) == (a.program is None):
        ap.error("give --submission or --program")
    if a.program and not a.challenge:
        ap.error("--program needs --challenge")
    if bridge.STANDALONE and a.videos != "none":
        ap.error("--videos: the CLI renders clips (duckbench simulate / share); here --videos none")
    out = Path(a.out) if a.out else duck_rl.HOME / "evals" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    # the TS side runs in another cwd (bridge.cwd() / the repo): a file opponent goes by its absolute path (a house ref
    # in any case, as the TS side's parseOpponent takes it; "stand" as it is)
    builtin = lambda o: o.strip().lower().startswith("house:") or o.strip() == "stand"  # noqa: E731
    opponents = [x for o in a.opponent for x in ("--opponent", o if builtin(o) else str(Path(o).resolve()))]
    with tempfile.TemporaryDirectory(dir=out) as td, budget.charged("eval", out.name, start=T0):
        sub = Path(a.submission).resolve() if a.submission else Path(td)
        if a.program:
            shutil.copy(resolve(a.program), sub / "program.json")
            pols = []
            for i, p in enumerate(a.policy):
                nodes, ref = p.split("=", 1)
                mv = moves.resolve_move(ref) if moves.is_library_ref(ref) else moves.resolve_path(Path(ref))
                shutil.copy(mv["json"], sub / f"p{i}.mlp.json")
                shutil.copy(mv["bin"], sub / f"p{i}.mlp.bin")
                pols.append({"nodes": nodes.split(","), "move": f"p{i}.mlp.json", "pin": mv["id"]})
            info = duck_rl.load_info(a.challenge)
            version = info["challenge"]["version"]
            meta = ({"format": moves.STRATEGY_FORMAT, "moves": pols} if version >= 1 else  # a spike Challenge (v0): the legacy format
                    {"format": "duck-rl-submission/1", "policies": [{"nodes": q["nodes"], "mlp": q["move"]} for q in pols]})
            (sub / "submission.json").write_text(json.dumps({
                **meta, "challenge": {"id": a.challenge, "version": version}, "program": "program.json"}))
        if bridge.STANDALONE:  # ts/eval.ts through the CLI, as eval_submission.py runs it (no videos, no result.md)
            r = bridge.run_ts("eval", ["--submission", sub, *(["--challenge", a.challenge] if a.challenge else []), "--seeds", a.seeds,
                                       *opponents, "--out", out / "summary.json"])
            if r.returncode == 0:
                s = json.loads((out / "summary.json").read_text())
                res = duel_line(s) if s.get("duel") is not None else solo_line(s)
                print(f"[eval] {res} -> {out / 'summary.json'}", flush=True)
            return r.returncode
        cmd = [sys.executable, str(duck_rl.REPO / "spikes/duck-rl/eval_submission.py"), "--submission", str(sub),
               "--seeds", a.seeds, "--out", str(out), "--videos", a.videos, *opponents]
        if a.challenge:
            cmd += ["--challenge", a.challenge]
        return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
