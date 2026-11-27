#!/usr/bin/env python3
"""T2' statistics: 4 arms x 3 seeds x {best,last} x 7 evaluation conditions.

Primary metric (GPT): V7 Last, C2 - C1 paired by training seed.
n=3 training runs; 32 evaluation seeds are an evaluation set, not repeats.
"""
import json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "r7/artifacts"
ARMS = ("c0", "c1", "c2", "c3")
CONDS = ("v1", "v2", "v3", "v4", "v5", "v6", "v7")
SEEDS = (301, 302, 303)
MISS = 30.0


def metrics(p):
    d = json.loads(p.read_text())
    runs = d["seeds"]
    t = sorted(r["result"] if r["finished"] else MISS for r in runs)
    h = len(t) // 2
    return dict(best_half=round(sum(t[:h]) / h, 4), mean=round(sum(t) / len(t), 4),
                finished=sum(1 for r in runs if r["finished"]), n=len(runs),
                falls=sum(r["falls"] for r in runs),
                eps_fall=sum(1 for r in runs if r["falls"]))


def main():
    out = {}
    for ck in ("best", "last"):
        for v in CONDS:
            for a in ARMS:
                rows = [metrics(ART / f"t2p_{a}_s{s}_{ck}_{v}.json") for s in SEEDS]
                out[f"{ck}/{v}/{a}"] = {k: round(st.mean(r[k] for r in rows), 4) for k in rows[0]}
                out[f"{ck}/{v}/{a}"]["per_seed"] = rows

    # primary: V7 last C2-C1 paired
    d = [out[f"last/v7/c2"]["per_seed"][i]["best_half"] - out[f"last/v7/c1"]["per_seed"][i]["best_half"] for i in range(3)]
    primary = dict(diffs=d, mean=round(sum(d) / 3, 4), n=3,
                   note="paired by training seed; 32 evaluation seeds per run")

    lines = []
    for ck in ("best", "last"):
        lines.append(f"\n### {ck}\n")
        lines.append("| arm | " + " | ".join(CONDS) + " |")
        lines.append("|---" * (len(CONDS) + 1) + "|")
        for a in ARMS:
            vals = " | ".join(f"{out[f'{ck}/{v}/{a}']['best_half']:.3f}" for v in CONDS)
            lines.append(f"| {a.upper()} | {vals} |")
        lines.append("")
        lines.append("完赛均值 / 32：")
        lines.append("| arm | " + " | ".join(CONDS) + " |")
        lines.append("|---" * (len(CONDS) + 1) + "|")
        for a in ARMS:
            vals = " | ".join(f"{out[f'{ck}/{v}/{a}']['finished']:.1f}" for v in CONDS)
            lines.append(f"| {a.upper()} | {vals} |")
    lines.append(f"\n主要指标：V7 Last 的 C2−C1 = {primary['mean']:+.4f} s（三 seed 差值 {['%+.3f' % x for x in d]}）")

    (ART / "t2p_summary.json").write_text(json.dumps(dict(groups=out, primary=primary), indent=1))
    (ROOT / "r7/T2P-TABLE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()