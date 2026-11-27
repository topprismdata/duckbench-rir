#!/usr/bin/env python3
"""E4-S1 statistics: 5 arms x 3 seeds, three checkpoints each (T0 / best / last).

Inputs: r4/artifacts/ts_{best,last}_a{arm}_s{seed}.json, ts_t0_a4_s{seed}.json and the
zero-training T0 of A0-A3 from r4/artifacts/t0_*_sim.json.
Output: r4/E4-TABLE.md + r4/artifacts/e4_summary.json
"""
import json, re, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "r4/artifacts"
ARMS = {0: "HFW", 1: "run-forward@2", 2: "hop-forward@2", 3: "sidestep-left@1", 4: "sidestep scratch"}
SEEDS = [107, 108, 109]
MISS = 30.0
T0_FILES = {0: "ROOT/r0/artifacts/sim_hfw.json", 1: "t0_base-run-forward_2_sim.json",
            2: "t0_base-hop-forward_2_sim.json", 3: "t0_base-sidestep-left_1_sim.json"}


def metrics(path):
    d = json.loads(path.read_text())
    runs = d["seeds"]
    times = sorted(r["result"] if r["finished"] else MISS for r in runs)
    half = len(times) // 2
    xs, ys = [], []
    for r in runs:
        m = re.search(r"end x(-?[\d.]+) y(-?[\d.]+)", r.get("timeline", ""))
        if m:
            xs.append(float(m.group(1))); ys.append(float(m.group(2)))
    return dict(best_half=round(sum(times[:half]) / half, 4),
                mean_all=round(sum(times) / len(times), 4),
                worst_half=round(sum(times[half:]) / (len(times) - half), 4),
                finished=sum(1 for r in runs if r["finished"]), n=len(runs),
                falls_events=sum(r["falls"] for r in runs),
                falls_episodes=sum(1 for r in runs if r["falls"]),
                x_mean=round(st.mean(xs), 3) if xs else None,
                x_max=round(max(xs), 3) if xs else None,
                y_mean=round(st.mean(ys), 3) if ys else None)


def main():
    rows = []
    for a in ARMS:
        entry = {"arm": a, "prior": ARMS[a]}
        if a in T0_FILES:
            entry["T0"] = metrics(ROOT / T0_FILES[a][5:]) if T0_FILES[a].startswith("ROOT/") else metrics(ART / T0_FILES[a])
        else:
            t0s = [metrics(ART / f"ts_t0_a4_s{s}.json") for s in SEEDS]
            entry["T0"] = {k: round(st.mean(t[k] for t in t0s), 4) for k in t0s[0]}
        for kind in ("best", "last"):
            per = [metrics(ART / f"ts_{kind}_a{a}_s{s}.json") for s in SEEDS]
            entry[kind] = {k: round(st.mean(p[k] for p in per), 4) for k in per[0]}
            entry[kind]["runs"] = per
        rows.append(entry)

    out = {"arms": rows, "seeds": SEEDS, "n_per_arm": len(SEEDS)}
    (ART / "e4_summary.json").write_text(json.dumps(out, indent=1))

    def line(e, kind):
        d = e[kind]
        return (f"| A{e['arm']} {e['prior']} | {d['best_half']:.4f} | {d['mean_all']:.4f} | {d['finished']}/{d['n']} | "
                f"{d['falls_events']}/{d['falls_episodes']} | {d['x_mean']} / {d['x_max']} | {d['y_mean']} |")

    L = ["| arm | best-half | mean | finished | falls events/episodes | x mean/max (m) | y mean (m) |",
         "|---|---|---|---|---|---|---|"]
    for kind, title in (("T0", "T0（训练前）"), ("best", "Best 导出"), ("last", "Last 检查点")):
        L.append(f"\n### {title}\n")
        L.append("| arm | best-half | mean | finished | falls events/episodes | x mean/max (m) | y mean (m) |")
        L.append("|---|---|---|---|---|---|---|")
        for e in rows:
            d = e[kind]
            L.append(f"| A{e['arm']} {e['prior']} | {d['best_half']:.4f} | {d['mean_all']:.4f} | {d['finished']}/{d['n']} | "
                     f"{d['falls_events']}/{d['falls_episodes']} | {d['x_mean']} / {d['x_max']} | {d['y_mean']} |")
    (ROOT / "r4/E4-TABLE.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()