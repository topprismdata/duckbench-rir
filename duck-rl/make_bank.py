#!/usr/bin/env python
"""Record a handoff bank: run a duckscript in the TS sim (stock policies, plus any trained ones you pass) and save, per
seed, the state at the first tick one of --nodes runs (with the draws, the loop state and the metric tail). Training
episodes can then start there (ENV start="handoff"). It is a few seconds of Mac time: run it as a charged run:

    budget.py run --designer <you> --label bank -- .venv/bin/python make_bank.py --challenge turntable \
        --program packages/challenges/baselines/turntable/clever.json --nodes backpedal --out banks/clever-backpedal.json

In a player's kit it charges the owner's budget itself (duck_rl/budget.py):

    uv run python make_bank.py --challenge wheel --program <duckscript.json> --nodes run --out home/banks/<name>.json

Programs resolve like train.py's (the kit, then the repo). Seeds default to 1000-1255 (never the eval seeds 0-31 / 100+).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

T0 = time.time()  # charged from here (the imports count)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import duck_rl  # noqa: E402,F401  (paths, the mode)
from duck_rl import bridge, budget  # noqa: E402

budget.begin("bank", T0)  # the budget record goes up before the heavy imports (train: torch, mujoco)

from train import resolve  # noqa: E402

ap = argparse.ArgumentParser()
_challenge = ap.add_mutually_exclusive_group(required=True)
_challenge.add_argument("--challenge")
_challenge.add_argument("--dare", dest="challenge", help=argparse.SUPPRESS)  # rename-challenge:keep: the pre-rename --dare, a silent alias
ap.add_argument("--program", required=True)
ap.add_argument("--nodes", required=True, help="walk/stand node ids (comma-separated): capture at the first of them")
ap.add_argument("--policy", action="append", default=[], help="nodes=path.mlp.json: trained policies swapped in")
ap.add_argument("--move", action="append", default=[],
                help='a moves entry as JSON (eval.ts --move), e.g. a raw prefix node: {"nodes": ["run"], "move": "/abs/x.mlp.json", '
                     '"host": "raw", "command": {"run": {"vx": 0.6}}}')
ap.add_argument("--seeds", default="1000-1255")
ap.add_argument("--out", required=True)
ap.add_argument("--full", action="store_true", help="run the full episodes (slower; only for the summary)")
a = ap.parse_args()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
args = ["--challenge", a.challenge, "--program", str(resolve(a.program)), "--seeds", a.seeds, "--bank", str(Path(a.out).resolve()),
        "--bank-nodes", a.nodes, "--quiet"]
for p in a.policy:
    nodes, path = p.split("=", 1)
    args += ["--policy", f"{nodes}={Path(path).resolve()}"]
for m in a.move:
    args += ["--move", m]
if a.full:
    args.append("--bank-full")
with budget.charged("bank", Path(a.out).name, start=T0):
    code = bridge.run_ts("eval", args).returncode
sys.exit(code)
