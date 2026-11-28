#!/usr/bin/env python3
"""Compare capacity arms: best-half, completion, mean x at t=2 s (from TS trace)."""
import json, statistics as st, sys
from pathlib import Path

def m(p):
    d = json.loads(Path(p).read_text())
    runs = d["seeds"]
    t = sorted(r["result"] if r["finished"] else 30.0 for r in runs)
    h = len(t) // 2
    x2 = [r["trace"][0][1] for r in runs if r.get("trace")]
    falls = sum(r["falls"] for r in runs)
    return dict(best_half=round(sum(t[:h]) / h, 3), mean=round(sum(t) / len(t), 3),
                finished=sum(1 for r in runs if r["finished"]), n=len(runs),
                x2=round(st.mean(x2), 3), falls=falls)

for p in sys.argv[1:]:
    print(p.split("/")[-1], m(p))
