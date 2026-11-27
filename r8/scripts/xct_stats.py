#!/usr/bin/env python3
"""R8-XCT statistics: 10 policies x 2 challenges x 2 commands.

Reports raw and normalized (vs steps-starter) metrics; never averages raw scores across
challenges. Paired comparisons C1-C0 and C2-C0 by training seed.
"""
import json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "r8/artifacts"
POLS = ["starter"] + [f"{a}_s{s}" for a in ("c0", "c1", "c2") for s in (301, 302, 303)]
CH = ("hurdles", "sprint")
VX = ("0p19", "0p10")
MISS = 30.0


def m(path):
    d = json.loads(path.read_text())
    runs = d["seeds"]
    t = sorted(r["result"] if r["finished"] else MISS for r in runs)
    h = len(t) // 2
    return dict(best_half=round(sum(t[:h]) / h, 4), mean=round(sum(t) / len(t), 4),
                worst_half=round(sum(t[h:]) / (len(t) - h), 4),
                finished=sum(1 for r in runs if r["finished"]), n=len(runs),
                falls=sum(r["falls"] for r in runs))


def main():
    data = {}
    for ch in CH:
        for vx in VX:
            for p in POLS:
                data[f"{ch}/{vx}/{p}"] = m(ART / f"xct_{p}_{ch}_{vx}.json")
    base = {f"{ch}/{vx}": data[f"{ch}/{vx}/starter"]["best_half"] for ch in CH for vx in VX}
    norm = {}
    for k, v in data.items():
        ch, vx, p = k.split("/")
        b = base[f"{ch}/{vx}"]
        norm[k] = round((v["best_half"] - b) / b, 4) if b else None

    lines = []
    for ch in CH:
        for vx in VX:
            lines.append(f"\n### {ch} vx {vx.replace('0p','0.')}\n")
            lines.append("| policy | best-half | mean | worst-half | finished | falls | rel vs starter |")
            lines.append("|---|---|---|---|---|---|---|")
            for p in POLS:
                d = data[f"{ch}/{vx}/{p}"]
                lines.append(f"| {p} | {d['best_half']:.3f} | {d['mean']:.3f} | {d['worst_half']:.3f} | "
                             f"{d['finished']}/32 | {d['falls']} | {norm[f'{ch}/{vx}/{p}']:+.3f} |")
    # paired C1-C0, C2-C0 per challenge/command on the ALL-SEED MEAN (the pre-registered family)
    paired = {}
    for ch in CH:
        for vx in VX:
            for arm in ("c1", "c2"):
                d = [data[f"{ch}/{vx}/{arm}_s{s}"]["mean"] - data[f"{ch}/{vx}/c0_s{s}"]["mean"] for s in (301, 302, 303)]
                paired[f"{ch}/{vx}/{arm}-c0"] = dict(diffs=[round(x, 4) for x in d], mean=round(st.mean(d), 4))
    # pre-registered primary endpoint: Hurdles 0.10, C1-C0, all-seed mean
    prim = paired["hurdles/0p10/c1-c0"]
    sd = st.stdev(prim["diffs"])
    t975 = 4.302652729911275  # t, df=2, two-sided 97.5%
    prim["sd"] = round(sd, 6)
    prim["ci95"] = [round(prim["mean"] - t975 * sd / 3 ** 0.5, 4), round(prim["mean"] + t975 * sd / 3 ** 0.5, 4)]
    # per-arm mean by challenge/command (for D and dD)
    means = {f"{ch}/{vx}/{p}": data[f"{ch}/{vx}/{p}"]["mean"] for ch in CH for vx in VX for p in POLS}
    # D = mean(0.10) - mean(0.19) per arm-policy-average, and dD = D(arm) - D(c0)
    D, dD = {}, {}
    for ch in CH:
        for arm in ("c0", "c1", "c2"):
            D[f"{ch}/{arm}"] = round(st.mean(means[f"{ch}/0p10/{arm}_s{s}"] for s in (301, 302, 303))
                                    - st.mean(means[f"{ch}/0p19/{arm}_s{s}"] for s in (301, 302, 303)), 4)
        for arm in ("c1", "c2"):
            dD[f"{ch}/{arm}-c0"] = round(st.mean(
                (means[f"{ch}/0p10/{arm}_s{s}"] - means[f"{ch}/0p19/{arm}_s{s}"])
                - (means[f"{ch}/0p10/c0_s{s}"] - means[f"{ch}/0p19/c0_s{s}"]) for s in (301, 302, 303)), 4)
            dD[f"{ch}/{arm}-c0_per_seed"] = [round(
                (means[f"{ch}/0p10/{arm}_s{s}"] - means[f"{ch}/0p19/{arm}_s{s}"])
                - (means[f"{ch}/0p10/c0_s{s}"] - means[f"{ch}/0p19/c0_s{s}"]), 4) for s in (301, 302, 303)]
    (ART / "xct_summary.json").write_text(json.dumps(dict(metrics=data, norm=norm, baseline=base, paired=paired,
                                                          means=means, D=D, dD=dD, primary=prim), indent=1))
    (ROOT / "r8/XCT-TABLE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(paired, indent=1))


if __name__ == "__main__":
    main()