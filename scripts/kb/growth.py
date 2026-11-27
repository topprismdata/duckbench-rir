#!/usr/bin/env python3
"""Growth metrics over the knowledge graph (philosophy dashboard).

Outputs kb/generated/GROWTH.md: evidence distribution, hypothesis throughput,
coverage per task, reuse of evidence in decisions, exclusion ledger.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "kb"


def load():
    ents = []
    for name in ("domain", "assets", "experiments", "evaluations", "knowledge"):
        p = KB / "data" / f"{name}.jsonl"
        if p.exists():
            ents += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rels = [json.loads(l) for l in (KB / "data/relations.jsonl").read_text().splitlines() if l.strip()]
    return ents, rels


def main():
    ents, rels = load()
    by_type = Counter(e["type"] for e in ents)
    claims = [e for e in ents if e["type"] == "Claim"]
    ev = Counter(c["data"].get("evidence_level") for c in claims)
    hyps = [e for e in ents if e["type"] == "Hypothesis"]
    hstat = Counter(h["data"].get("status") for h in hyps)
    exclusions = [e for e in ents if e["type"] == "Exclusion"]
    decisions = [e for e in ents if e["type"] == "Decision"]
    dec_edges = Counter(r["relation"] for r in rels if r["from"] in {d["id"] for d in decisions})
    supported_claims = {r["to"] for r in rels if r["relation"] in ("SUPPORTS", "REFUTES")}
    orphan_claims = [c["id"] for c in claims if c["id"] not in supported_claims]
    tasks = [e["id"] for e in ents if e["type"] == "Task"]
    cov = defaultdict(lambda: {"claims": 0, "hypotheses": 0})
    for c in claims:
        sc = c["data"].get("scope", "")
        for t in tasks:
            if t.split(":")[2] in sc:
                cov[t]["claims"] += 1
    for h in hyps:
        for t in tasks:
            if t.split(":")[2] in json.dumps(h["data"], ensure_ascii=False):
                cov[t]["hypotheses"] += 1

    L = ["# Growth dashboard (generated)", "",
         f"entities={len(ents)} relations={len(rels)} tasks={len(tasks)} claims={len(claims)} hypotheses={len(hyps)}", "",
         "## Evidence distribution", ""]
    for k, v in ev.most_common():
        L.append(f"- {k}: {v}")
    L += ["", "## Hypothesis status", ""]
    for k, v in hstat.most_common():
        L.append(f"- {k}: {v}")
    L += ["", "## Coverage per task", ""]
    for t, c in sorted(cov.items()):
        L.append(f"- {t}: claims {c['claims']}, hypotheses {c['hypotheses']}")
    L += ["", "## Exclusions (reopen condition)", ""]
    for e in exclusions:
        L.append(f"- {e['id']}: {e['data']['reopen_condition']}")
    L += ["", "## Decision evidence use", ""]
    for k, v in dec_edges.most_common():
        L.append(f"- {k}: {v} edges")
    L += ["", "## Orphan claims (no SUPPORTS/REFUTES)", ""]
    L += [f"- {x}" for x in orphan_claims] or ["- none"]

    out = KB / "generated" / "GROWTH.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()