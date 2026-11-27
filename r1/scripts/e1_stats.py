#!/usr/bin/env python3
"""E1 statistics (v2, after GPT R1 review).

Fixes:
 - ranked times use the TS `result` field (3 dp) instead of the 2 dp `finish_t`;
 - fall counting separates episodes from events and splits pre/post finish;
 - host load average and step throughput per run are reported (resource contention).

Inputs: r1/artifacts/e1_ts_s{1..5}.json, e1_warm_start.json
        duck-rl/home/runs/e1_s*/{summary.json,log.jsonl}
Output: r1/E1-TABLE.md, r1/artifacts/e1_summary.json
"""
import json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "r1/artifacts"
RUNS = ROOT / "duck-rl/home/runs"
MISS = 30.0


def load(p):
    return json.loads(p.read_text())


def fall_split(runs):
    """Episodes vs events, split by whether the FIRST fall was before the finish.

    Event-level timestamps are not in the TS JSON, so an episode whose first fall
    precedes its finish has an unattributable split; counted separately."""
    out = dict(episodes_with_pre_finish_fall=0, episodes_with_post_finish_fall=0,
               episodes_first_fall_unfinished=0, episodes_mixed_unattributable=0,
               pre_finish_fall_events=0, post_finish_fall_events=0, mixed_fall_events=0)
    for r in runs:
        if not r["falls"]:
            continue
        if not r["finished"]:
            out["episodes_first_fall_unfinished"] += 1
            out["mixed_fall_events"] += r["falls"]
        elif r["first_fall_t"] is not None and r["first_fall_t"] <= r["finish_t"]:
            out["episodes_with_pre_finish_fall"] += 1
            out["mixed_fall_events"] += r["falls"]
            out["pre_finish_fall_events"] += 1  # at least this one is pre-finish
        else:
            out["episodes_with_post_finish_fall"] += 1
            out["post_finish_fall_events"] += r["falls"]
    return out


def host(s):
    d = load(RUNS / f"e1_s{s}" / "summary.json")
    loads, waits = [], []
    for line in (RUNS / f"e1_s{s}" / "log.jsonl").read_text().splitlines():
        try:
            it = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "loadavg1" in it:
            loads.append(it["loadavg1"])
        if "t_wait" in it:
            waits.append(it["t_wait"])
    return {
        "steps": d["steps"], "iters": d["iters"], "train_s": d["train_s"],
        "sps": round(d["steps"] / d["train_s"]),
        "load_median": round(st.median(loads), 2) if loads else None,
        "load_max": round(max(loads), 2) if loads else None,
        "t_wait_total_s": round(sum(waits), 1) if waits else None,
    }


def stat(seed):
    d = load(ART / f"e1_ts_s{seed}.json")
    runs = d["seeds"]
    times = [r["result"] if r["finished"] else MISS for r in runs]
    fs = sorted(times)
    half = len(fs) // 2
    row = {
        "seed": seed,
        "ranked_best_half": round(sum(fs[:half]) / half, 7),
        "mean_all": round(sum(fs) / len(fs), 7),
        "worst_half": round(sum(fs[half:]) / (len(fs) - half), 7),
        "worst": round(max(fs), 4),
        "best": round(min(fs), 4),
        "finished": sum(1 for r in runs if r["finished"]),
        "n": len(runs),
    }
    row.update(fall_split(runs))
    row.update(host(seed))
    return row


def main():
    rows = [stat(s) for s in range(1, 6)]
    ws = load(ART / "e1_warm_start.json")["seeds"]
    wt = sorted(r["result"] if r["finished"] else MISS for r in ws)
    warm = {
        "ranked_best_half": round(sum(wt[:16]) / 16, 7),
        "mean_all": round(sum(wt) / len(wt), 7),
        "worst_half": round(sum(wt[16:]) / 16, 7),
        "worst": round(max(wt), 4),
        "finished": sum(1 for r in ws if r["finished"]),
        "n": len(ws),
    }
    warm.update(fall_split(ws))
    bh = [r["ranked_best_half"] for r in rows]
    sd = st.stdev(bh)
    tot = {k: sum(r[k] for r in rows) for k in
           ("episodes_with_pre_finish_fall", "episodes_with_post_finish_fall",
            "pre_finish_fall_events", "post_finish_fall_events",
            "episodes_mixed_unattributable", "finished", "n")}
    summary = {"runs": rows, "warm_start": warm, "best_half_sd": round(sd, 7),
               "best_half_range": round(max(bh) - min(bh), 7),
               "mean_of_runs": round(sum(bh) / len(bh), 7), "totals": tot}
    (ART / "e1_summary.json").write_text(json.dumps(summary, indent=1))

    L = ["| run | best-half | mean(all) | worst-half | worst | finished | falls pre/post (events) | episodes w/ fall | load1 median/max | steps/s |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| e1_s{r['seed']} | {r['ranked_best_half']:.4f} | {r['mean_all']:.4f} | {r['worst_half']:.4f} | "
                 f"{r['worst']:.2f} | {r['finished']}/{r['n']} | {r['pre_finish_fall_events']}/{r['post_finish_fall_events']} | "
                 f"{r['episodes_with_pre_finish_fall']}/{r['episodes_with_post_finish_fall']} | {r['load_median']}/{r['load_max']} | {r['sps']} |")
    L.append(f"| warm_start | {warm['ranked_best_half']:.4f} | {warm['mean_all']:.4f} | {warm['worst_half']:.4f} | "
             f"{warm['worst']:.2f} | {warm['finished']}/{warm['n']} | {warm['pre_finish_fall_events']}/{warm['post_finish_fall_events']} | "
             f"{warm['episodes_with_pre_finish_fall']}/{warm['episodes_with_post_finish_fall']} | — | — |")
    notes = [
        f"- 五次 run 平均 best-half: {summary['mean_of_runs']:.4f} s（相对 warm start 缩短 {(1 - summary['mean_of_runs']/warm['ranked_best_half'])*100:.1f} %）",
        f"- run 间 best-half sd: {sd:.4f} s；极差 {summary['best_half_range']:.4f} s（n=5）",
        f"- 合计：完赛 {tot['finished']}/{tot['n']}；冲线前跌倒事件 {tot['pre_finish_fall_events']}；"
        f"冲线后跌倒事件 {tot['post_finish_fall_events']}；不可归因事件 {tot['mixed_fall_events'] if 'mixed_fall_events' in tot else 0}",
        f"- 冲线后发生跌倒的 episode: {tot['episodes_with_post_finish_fall']}/{tot['n']}"
        f"（{tot['episodes_with_post_finish_fall']/tot['n']*100:.1f} %）",
    ]
    (ROOT / "r1/E1-TABLE.md").write_text("\n".join(L) + "\n\n" + "\n".join(notes) + "\n")
    print("\n".join(L))
    print("\n".join(notes))


if __name__ == "__main__":
    main()