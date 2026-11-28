#!/usr/bin/env python3
"""Newcomer simulation: shortest learning path to a target task.

  python scripts/kb/path.py --task hurdles

Returns: competencies required by the task's objectives (or by claims about the task),
the activities that develop them, the resources they use, and the evidence that would
close the loop. Missing links are reported as GAPS, not invented.
"""
import argparse, json
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "kb"


def load():
    ents = []
    for p in sorted((KB / "data").glob("*.jsonl")):
        if p.name == "relations.jsonl":
            continue
        ents += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rels = [json.loads(l) for l in (KB / "data/relations.jsonl").read_text().splitlines() if l.strip()]
    return ents, rels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="task key, e.g. hurdles")
    a = ap.parse_args()
    ents, rels = load()
    by_id = {e["id"]: e for e in ents}
    out_e = defaultdict(list)
    in_e = defaultdict(list)
    for r in rels:
        out_e[r["from"]].append(r)
        in_e[r["to"]].append(r)

    task = next((e for e in ents if e["type"] == "Task" and f":{a.task}:" in e["id"]), None)
    if not task:
        print("unknown task", a.task); return
    scope = next((e["id"] for e in ents if e["type"] == "Scope" and f":{a.task}:" in e["id"]), None)

    # competencies required by objectives that mention the task via their activities' scope
    comps = []
    for r in out_e.get(scope, []):
        pass
    objs = [e for e in ents if e["type"] == "LearningObjective"]
    for o in objs:
        if any(r["from"] == o["id"] and r["relation"] == "REQUIRES_FOR_GOAL" for r in rels):
            comps += [r["to"] for r in rels if r["from"] == o["id"] and r["relation"] == "REQUIRES_FOR_GOAL"]
    comps = sorted(set(comps))

    print(f"target: {task['id']}")
    print(f"scope:  {scope}")
    print(f"required competencies: {len(comps)}")
    for c in comps:
        ce = by_id.get(c, {})
        acts = [r["from"] for r in in_e.get(c, []) if r["relation"] == "DEVELOPS_COMPETENCY"]
        ress = [r["from"] for r in in_e.get(c, []) if r["relation"] == "TEACHES_COMPETENCY"]
        cas = [r["from"] for r in in_e.get(c, []) if r["relation"] == "ASSESSES_COMPETENCY"]
        print(f"- {c}: {ce.get('data',{}).get('statement','?')}")
        print(f"    activities: {acts or 'GAP'}")
        print(f"    resources:  {ress or 'GAP'}")
        print(f"    assessments:{cas or 'GAP — competency not demonstrated yet'}")
    # claims covering the task
    claims = [e["id"] for e in ents if e["type"] == "Claim" and scope and e["data"].get("scope") == scope]
    print(f"\nclaims covering this task: {claims or 'GAP'}")
    hyps = [e["id"] for e in ents if e["type"] == "Hypothesis" and a.task in json.dumps(e["data"], ensure_ascii=False)]
    print(f"hypotheses touching it:    {hyps or 'none'}")
    ex = [e["id"] for e in ents if e["type"] == "Exclusion" and a.task in e["id"]]
    print(f"exclusions in force:       {ex or 'none'}")


if __name__ == "__main__":
    main()