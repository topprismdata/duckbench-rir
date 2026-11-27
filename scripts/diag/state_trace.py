#!/usr/bin/env python3
"""Trace challenge state columns through the Python env (diagnostics, not a score).

  python scripts/diag/state_trace.py --config configs/wheel/baseline.py \
      --mlp submissions/r17_w741/policy.mlp.json --n 8 --out r22/wheel_trace.jsonl
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "duck-rl"
sys.path.insert(0, str(KIT))

import numpy as np  # noqa: E402
import duck_rl  # noqa: E402
from duck_rl.env import load_bank, make_env, make_unpack  # noqa: E402
from duck_rl.simd_mlp import SimdMlp  # noqa: E402
from train import compile_program, config_challenge, env_config, load_config, resolve  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--mlp", default=None)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--out", required=True)
a = ap.parse_args()

cfg_path = (KIT / a.config).resolve() if not Path(a.config).is_absolute() else Path(a.config)
C = load_config(cfg_path)
challenge = config_challenge(C, cfg_path)
compiled = compile_program(challenge, resolve(C.PROGRAM), duck_rl.HOME / "tmp" / "diag-compiled.json")
cfg = env_config(C, challenge, compiled)
info = cfg["info"]
unpack = make_unpack(info)
pol = SimdMlp(Path(a.mlp) if a.mlp else duck_rl.PLANT / "policies" / "alpha_walking.mlp.json")
bank = load_bank(cfg["bank"]) if cfg.get("start") == "handoff" and cfg.get("bank") else []

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
env = make_env(cfg, 7)
with out.open("w") as f:
    for i in range(min(a.n, len(bank)) if bank else a.n):
        obs, _ = env.reset(bank_state=bank[i]) if bank else env.reset()
        S = []
        while True:
            act = np.asarray(pol(obs.astype(np.float32)), np.float32).reshape(14)
            obs, _, st, term, to = env.step(act)
            S.append(st)
            if term or to:
                break
        D = unpack(np.stack(S))
        keep = {k: np.asarray(v, dtype=float).round(5).tolist() for k, v in D.items()
                if np.ndim(v) == 1 and len(v) == len(S)}
        f.write(json.dumps({"episode": i, "metric": float(env.score), "columns": keep}) + "\n")
print(f"wrote {out} ({i+1} episodes, columns {len(keep)})")