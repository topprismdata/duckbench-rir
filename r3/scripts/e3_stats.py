#!/usr/bin/env python3
"""E3 paired statistics: A (warm-start) vs B (scratch), 6 pairs, TS seeds 8-39.

d_i = A_i - B_i on the ranked best-half (negative = warm-start faster).
Reports per-run metrics, the six paired differences, mean/SD/95% t interval, and completion
rates with the 30 s miss penalty. No equivalence claim is made when the interval spans 0.
"""
import json, math, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "r3/artifacts"
MISS = 30.0
SEEDS = [101, 102, 103, 104, 105, 106]
T975 = {5: 2.570582}  # two-sided 97.5% t, df=5


def load(path):
    return json.loads(path.read_text())


def metrics(path):
    d = load(path)
    runs = d["seeds"]
    times = sorted(r["result"] if r["finished"] else MISS for r in runs)
    half = len(times) // 2
    pre = sum(1 for r in runs if r["falls"] and r["first_fall_t"] is not None and r["finished"] and r["first_fall_t"] <= r["finish_t"])
    post = sum(1 for r in runs if r["falls"] and r["finished"] and r["first_fall_t"] is not None and r["first_fall_t"] > r["finish_t"])
    events_pre = sum(r["falls"] for r in runs if r["falls"] and r["finished"] and r["first_fall_t"] is not None and r["first_fall_t"] <= r["finish_t"])
    events_post = sum(r["falls"] for r in runs if r["falls"] and r["finished"] and r["first_fall_t"] is not None and r["first_fall_t"] > r["finish_t"])
    return dict(best_half=sum(times[:half]) / half, mean_all=sum(times) / len(times),
                worst_half=sum(times[half:]) / (len(times) - half), worst=max(times),
                finished=sum(1 for r in runs if r["finished"]), n=len(runs),
                episodes_pre_fall=pre, episodes_post_fall=post,
                events_pre_fall=events_pre, events_post_fall=events_post)


def main():
    rows, diffs = [], []
    for s in SEEDS:
        a, b = metrics(ART / f"ts_a{s}.json"), metrics(ART / f"ts_b{s}.json")
        d = a["best_half"] - b["best_half"]
        diffs.append(d)
        rows.append(dict(seed=s, A=a, B=b, d_a_minus_b=d))
    n = len(diffs)
    mean = sum(diffs) / n
    sd = st.stdev(diffs)
    se = sd / math.sqrt(n)
    half = T975[n - 1] * se
    out = dict(
        pairs=rows, n=n, mean_d=mean, sd_d=sd, se=se,
        ci95=[mean - half, mean + half],
        sesoi_reference_s=0.2,
        decision=("warm-start faster beyond reference" if mean + half < -0.2 else
                  "warm-start faster, practical size unclear" if mean + half < 0 else
                  "direction unclear (CI spans 0)" if mean - half < 0 else
                  "scratch faster"),
        note="paired differences; CI from t(df=5); a CI spanning 0 is not evidence of equivalence",
    )
    (ART / "e3_summary.json").write_text(json.dumps(out, indent=1))
    L = ["| pair | A best-half | B best-half | d = A−B | A fin | B fin |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['seed']} | {r['A']['best_half']:.4f} | {r['B']['best_half']:.4f} | {r['d_a_minus_b']:+.4f} | "
                 f"{r['A']['finished']}/32 | {r['B']['finished']}/32 |")
    L.append("")
    L.append(f"- d 均值 {mean:+.4f} s；sd {sd:.4f}；95% CI [{mean-half:+.4f}, {mean+half:+.4f}]")
    L.append(f"- 裁决（按 0.2 s 参考）：{out['decision']}")
    (ROOT / "r3/E3-TABLE.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()