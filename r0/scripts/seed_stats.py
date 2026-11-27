#!/usr/bin/env python3
"""R0-4 evidence: per-seed statistics of two sprint baselines.

Usage: python3 r0/scripts/seed_stats.py
Reads r0/artifacts/sim_hfw.json and sim_stock.json (produced by
`duckbench simulate sprint <file> --seeds 8-39 --json`).
"""
import json, random, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    j = json.loads((ROOT / "r0" / "artifacts" / name).read_text())
    return j, {s["seed"]: (s["result"] if s.get("finished") else 30.0) for s in j["seeds"]}


def best_half(xs):
    xs = sorted(xs)
    k = len(xs) // 2
    return sum(xs[:k]) / k


def main():
    random.seed(0)
    jh, a = load("sim_hfw.json")
    js, b = load("sim_stock.json")
    for name, d in (("hurdles-fast-walker@1", a), ("stock walk", b)):
        xs = [d[k] for k in sorted(d)]
        print(f"{name}: n={len(xs)} mean={st.mean(xs):.3f} sd={st.stdev(xs):.3f} "
              f"min={min(xs):.3f} max={max(xs):.3f} best_half={best_half(xs):.3f}")

    seeds = sorted(set(a) & set(b))
    diffs = [b[s] - a[s] for s in seeds]
    print(f"paired stock-hfw: mean={st.mean(diffs):.3f} sd={st.stdev(diffs):.3f} n={len(diffs)}")

    boot = []
    for _ in range(2000):
        idx = [random.choice(seeds) for _ in seeds]
        boot.append(best_half([b[s] for s in idx]) - best_half([a[s] for s in idx]))
    boot.sort()
    print(f"bootstrap 95% CI of best-half diff: {boot[50]:.4f} .. {boot[1949]:.4f}")

    vals = []
    for _ in range(5000):
        idx = random.sample(seeds, 24)
        vals.append(best_half([a[s] for s in idx]))
    vals.sort()
    print(f"hfw hidden-24 best-half: mean={st.mean(vals):.4f} sd={st.stdev(vals):.4f} "
          f"p2.5={vals[125]:.3f} p97.5={vals[4874]:.3f}")
    print("run ids:", jh["run_id"], js["run_id"])


if __name__ == "__main__":
    main()