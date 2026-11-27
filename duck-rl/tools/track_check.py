"""How well a base move follows its command (kit 1.17.0; configs/base/move.py): runs the exported move in the training
env (the same physics and noise as training) at fixed commands and prints, per command, the trunk's mean forward speed
vx, sideways speed vy and turn rate wz over the episodes (after the first second), and the falls.

    .venv/bin/python tools/track_check.py --submission submissions/my-turn-left [--set BASE_MOVE=turn_left] \\
        [--command "wz=1.0" --command "wz=2.0"] [--episodes 8] [--config configs/base/move.py]

Without --command it checks the preset's command (RAW_COMMAND) and the two ends of its ranges (COMMAND_RANGES).
Python-env numbers (not the TS sim's): use them to compare your base moves; `duckbench simulate` shows the move itself.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import train  # noqa: E402  (the config loader, env_config, compile_program)
from duck_rl.controller import COMMAND_SLOTS  # noqa: E402
from duck_rl.duel import load_runner  # noqa: E402
from duck_rl.env import make_env  # noqa: E402
from lab.duck import LAYOUT  # noqa: E402

SLOTS = ("vx", "vy", "wz")


def parse_command(text: str) -> dict:
    out = {}
    for part in text.replace(";", ",").split(","):
        if part.strip():
            k, _, v = part.partition("=")
            if k.strip() not in SLOTS:
                raise SystemExit(f"--command {text!r}: slots are {', '.join(SLOTS)}")
            out[k.strip()] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--submission", required=True, help="submissions/<label> (its policy.mlp.json)")
    ap.add_argument("--config", default=str(HERE / "configs/base/move.py"))
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--command", action="append", default=[], help='e.g. "vx=0.3,wz=0.8" (unset slots: 0)')
    ap.add_argument("--episodes", type=int, default=8)
    a = ap.parse_args()

    cfg_path = Path(a.config).resolve()
    cfg_mod = train.load_config(cfg_path)
    train.apply_overrides(cfg_mod, a.sets)
    if callable(getattr(cfg_mod, "after_set", None)):
        cfg_mod.after_set()
    challenge = train.config_challenge(cfg_mod, cfg_path)
    nodes = train.trained_nodes(cfg_mod)
    compiled = train.compile_program(challenge, train.resolve(cfg_mod.PROGRAM), HERE / "home" / "track_check" / "compiled.json")
    env_cfg = train.env_config(cfg_mod, challenge, compiled)
    sub = Path(a.submission)
    mlp = sub / "policy.mlp.json" if sub.is_dir() else sub
    runner = load_runner(mlp.resolve())

    if a.command:
        commands = [parse_command(c) for c in a.command]
    else:
        base = {k: float(v) for k, v in cfg_mod.RAW_COMMAND[nodes[0]].items() if k in SLOTS}
        ranges = {k: r for k, r in (getattr(cfg_mod, "COMMAND_RANGES", {}) or {}).items() if k in SLOTS}
        commands = [base] + [dict(base, **{k: r[i] for k, r in ranges.items()}) for i in (0, 1)] if ranges else [base]
    lv, av = LAYOUT["lin_vel"][0], LAYOUT["ang_vel"][0]
    fell_i = LAYOUT["fell"][0]
    print(f"{mlp}: {len(commands)} command(s) x {a.episodes} episodes of {env_cfg.get('end_s', '?')} s (Python env)")
    print(f"{'command':<28} {'vx m/s':>8} {'vy m/s':>8} {'wz rad/s':>9} {'falls':>6}")
    for cmd in commands:
        c = dict(env_cfg)
        # every slot fixed at the command (a range of one value): the training reset's other draws and noise stay
        c["command_ranges"] = {str(COMMAND_SLOTS.index(k)): (float(cmd.get(k, 0.0)),) * 2 for k in SLOTS}
        env = make_env(c, 7)
        rows, falls = [], 0
        act = np.zeros(14, np.float32)
        for _ in range(a.episodes):
            obs, _ = env.reset()
            k = 0
            while True:
                runner.run(obs, act)
                obs, _, st, term, trunc = env.step(act.copy())
                k += 1
                if k > 50:  # after the first second (the start)
                    rows.append((st[lv], st[lv + 1], st[av + 2]))
                if term or trunc:
                    falls += int(st[fell_i] > 0)
                    break
        m = np.mean(rows, axis=0) if rows else (np.nan,) * 3
        label = ", ".join(f"{k}={cmd.get(k, 0.0):g}" for k in SLOTS)
        print(f"{label:<28} {m[0]:>8.3f} {m[1]:>8.3f} {m[2]:>9.3f} {falls:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
