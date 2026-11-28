#!/usr/bin/env python3
"""Validate the DuckBench research knowledge graph (kb/data/*.jsonl) against kb/ontology.

Exit 0 = all gates pass. Errors are printed with the entity/relation id.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "kb"
ONTO = json.loads((KB / "ontology/ontology.json").read_text())
TYPES = ONTO["types"]
RELS = set(ONTO["relations"])
COMMON = ["id", "type", "schema_version", "label", "recorded_at", "created_by", "origin", "source_refs", "data"]

errors, warnings = [], []


def load(name):
    p = KB / "data" / f"{name}.jsonl"
    out = []
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if line.strip():
            out.append(json.loads(line))
    return out


def main():
    ents, rels = [], []
    for name in ("domain", "assets", "experiments", "evaluations", "knowledge", "learning"):
        ents += load(name)
    rels = [json.loads(l) for l in (KB / "data/relations.jsonl").read_text().splitlines() if l.strip()]

    ids = {}
    for e in ents:
        for f in COMMON:
            if f not in e:
                errors.append(f"{e.get('id','?')}: missing field {f}")
        eid, typ = e.get("id", ""), e.get("type")
        if typ not in TYPES:
            errors.append(f"{eid}: unknown type {typ}")
            continue
        pref = TYPES[typ]["prefix"]
        if not eid.startswith(pref + ":"):
            errors.append(f"{eid}: id prefix should be {pref}:")
        if eid in ids:
            errors.append(f"{eid}: duplicate id")
        ids[eid] = e
        for f in TYPES[typ]["required"]:
            if f not in e.get("data", {}):
                errors.append(f"{eid}: data.{f} missing (required by {typ})")
        for s in e.get("source_refs", []):
            if s.get("kind") == "REPOSITORY_FILE" and s.get("path"):
                if not (ROOT / s["path"]).exists():
                    warnings.append(f"{eid}: source_ref path missing {s['path']}")

    for r in rels:
        a, b, rel = r.get("from"), r.get("to"), r.get("relation")
        if rel not in RELS:
            errors.append(f"relation {a} -{rel}-> {b}: unknown relation")
        if a not in ids:
            errors.append(f"relation from unknown {a}")
        if b not in ids:
            errors.append(f"relation to unknown {b}")

    # closure gates
    sups = {r["to"] for r in rels if r["relation"] in ("SUPPORTS", "REFUTES")}
    tested = {r["to"] for r in rels if r["relation"] == "TESTED_BY"}
    for e in ents:
        if e["type"] == "Claim":
            if e["id"] not in sups and not e.get("source_refs"):
                errors.append(f"{e['id']}: claim has no SUPPORTS edge and no source_refs")
        if e["type"] == "Hypothesis":
            st = e["data"].get("status")
            if st in ("SUPPORTED", "FALSIFIED") and e["id"] not in tested:
                errors.append(f"{e['id']}: closed hypothesis has no TESTED_BY")
        if e["type"] == "Exclusion":
            rr = e["data"].get("reason_ref")
            if rr and rr not in ids:
                errors.append(f"{e['id']}: reason_ref {rr} unknown")
        if e["type"] == "MetricResult":
            if not any(r["from"] == e["id"] and r["relation"] == "EVALUATED_ON" for r in rels):
                errors.append(f"{e['id']}: MetricResult without EVALUATED_ON")
        if e["type"] == "CompetencyAssessment":
            for need in ("DEMONSTRATED_BY", "FOLLOWS_ASSESSMENT_SPEC", "ASSESSES_LEARNER"):
                if not any(r["from"] == e["id"] and r["relation"] == need for r in rels):
                    errors.append(f"{e['id']}: CompetencyAssessment missing {need}")
        if e["type"] == "LearningActivity":
            if not any(r["from"] == e["id"] and r["relation"] == "PERFORMED_BY" for r in rels):
                errors.append(f"{e['id']}: LearningActivity missing PERFORMED_BY")
            if e["data"].get("outcome") not in ("SUCCESS", "PARTIAL", "FAILED", "ABANDONED"):
                errors.append(f"{e['id']}: invalid outcome")
        if e["type"] == "TransferAssessment":
            if not any(r["from"] == e["id"] and r["relation"] == "TRANSFERS_ASSET" for r in rels):
                warnings.append(f"{e['id']}: TransferAssessment without TRANSFERS_ASSET")
        if e["type"] == "CompetitionAttempt":
            if e["data"].get("rank") == 1 and not any(
                    r["from"] == e["id"] and r["relation"] == "SUPPORTS" for r in rels):
                warnings.append(f"{e['id']}: champion attempt not linked to a claim")

    for w in warnings:
        print("WARN", w)
    for err in errors:
        print("ERROR", err)
    print(f"entities={len(ents)} relations={len(rels)} warnings={len(warnings)} errors={len(errors)}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()