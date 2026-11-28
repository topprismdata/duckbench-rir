#!/usr/bin/env python3
"""Build kb/generated/graph.sqlite and kb/generated/GRAPH.md from kb/data/*.jsonl."""
import json, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "kb"
GEN = KB / "generated"
GEN.mkdir(exist_ok=True)


def load():
    ents = []
    for name in ("domain", "assets", "experiments", "evaluations", "knowledge", "learning"):
        for line in (KB / "data" / f"{name}.jsonl").read_text().splitlines():
            if line.strip():
                ents.append(json.loads(line))
    rels = [json.loads(l) for l in (KB / "data/relations.jsonl").read_text().splitlines() if l.strip()]
    return ents, rels


def main():
    ents, rels = load()
    db = GEN / "graph.sqlite"
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE entities (id TEXT PRIMARY KEY, type TEXT, label TEXT, data TEXT)")
    con.execute("CREATE TABLE relations (src TEXT, rel TEXT, dst TEXT)")
    con.executemany("INSERT INTO entities VALUES (?,?,?,?)",
                    [(e["id"], e["type"], e["label"], json.dumps(e["data"], ensure_ascii=False)) for e in ents])
    con.executemany("INSERT INTO relations VALUES (?,?,?)", [(r["from"], r["relation"], r["to"]) for r in rels])
    con.commit()

    by_type = {}
    for e in ents:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    claims = [e for e in ents if e["type"] == "Claim"]
    hyps = [e for e in ents if e["type"] == "Hypothesis"]
    exclusions = [e for e in ents if e["type"] == "Exclusion"]

    def support(cid):
        arts = [r["from"] for r in rels if r["to"] == cid and r["relation"] in ("SUPPORTS", "REFUTES")]
        names = []
        for a in arts:
            ent = next((x for x in ents if x["id"] == a), None)
            if ent:
                names.append(f"{a} ({ent['type']}: {ent['label']})")
        return names

    L = ["# Knowledge graph (generated — do not edit)", "",
         f"Entities: {len(ents)} · relations: {len(rels)} · types: {len(by_type)}", "",
         "## Claims and their evidence", ""]
    for c in sorted(claims, key=lambda x: x["id"]):
        L.append(f"- **{c['id']}** — {c['label']} *({c['data'].get('evidence_level')})*")
        for s in support(c["id"]):
            L.append(f"  - ← {s}")
    L += ["", "## Hypotheses", ""]
    for h in sorted(hyps, key=lambda x: x["id"]):
        L.append(f"- {h['id']} [{h['data'].get('status')}] {h['data'].get('statement')}")
    L += ["", "## Exclusions (do not repeat without reopening condition)", ""]
    for e in sorted(exclusions, key=lambda x: x["id"]):
        L.append(f"- {e['id']}: {e['data'].get('route')} — reopen if {e['data'].get('reopen_condition')}")
    L += ["", "## Types", ""]
    for t, n in sorted(by_type.items()):
        L.append(f"- {t}: {n}")
    (GEN / "GRAPH.md").write_text("\n".join(L) + "\n")
    print(f"built {db} and GRAPH.md ({len(ents)} entities, {len(rels)} relations)")


if __name__ == "__main__":
    main()