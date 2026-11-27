#!/usr/bin/env python3
"""E2.1 statistics: 3 arms x 3 seeds x {best,last} x 3 evaluation commands.

d = arm - T0 on the same seed/checkpoint/condition, paired over the 3 training seeds.
n=3 is exploratory only (GPT DEC-05).
"""
import json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "r5/artifacts"
ARMS = ("t0", "t1", "t2")
CONDS = ("fixed19", "fixed10", "staged")
SEEDS = (201, 202, 203)
MISS = 30.0


def metrics(p):
    d = json.loads(p.read_text())
    runs = d["seeds"]
    t = sorted(r["result"] if r["finished"] else MISS for r in runs)
    h = len(t) // 2
    return dict(best_half=round(sum(t[:h]) / h, 4), mean=round(sum(t) / len(t), 4),
                worst_half=round(sum(t[h:]) / (len(t) - h), 4),
                finished=sum(1 for r in runs if r["finished"]), n=len(runs),
                falls=sum(r["falls"] for r in runs),
                eps_fall=sum(1 for r in runs if r["falls"]))


def main():
    out = {}
    for ck in ("best", "last"):
        for cond in CONDS:
            for a in ARMS:
                rows = [metrics(ART / f"e21_{a}_s{s}_{ck}_{cond}.json") for s in SEEDS]
                out[f"{ck}/{cond}/{a}"] = {k: round(st.mean(r[k] for r in rows), 4) for k in rows[0]}
                out[f"{ck}/{cond}/{a}"]["per_seed"] = rows
    lines = []
    for ck in ("best", "last"):
        lines.append(f"\n### {ck}\n")
        lines.append("| arm | fixed 0.19 | fixed 0.10 | staged 0.10→0.19 | finished mean (19/10/staged, of 32) | falls (19/10/staged) |")
        lines.append("|---|---|---|---|---|---|")
        for a in ARMS:
            c = [out[f"{ck}/{cond}/{a}"] for cond in CONDS]
            lines.append(f"| {a.upper()} | " + " | ".join(f"{x['best_half']:.4f} s" for x in c) +
                         f" | {c[0]['finished']:.1f}/{c[1]['finished']:.1f}/{c[2]['finished']:.1f} of 32 | "
                         f"{c[0]['falls']:.0f}/{c[1]['falls']:.0f}/{c[2]['falls']:.0f} |")
    (ART / "e21_summary.json").write_text(json.dumps(out, indent=1))
    (ROOT / "r5/E21-TABLE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()