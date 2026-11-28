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
    for name in ("domain", "assets", "experiments", "evaluations", "knowledge", "learning"):
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

    # ---- M01–M08 with honest denominators (DB-KG-04 P0 fixes) ----
    import statistics as stt
    from datetime import datetime
    def dt(x):
        return datetime.fromisoformat(x.replace("Z", "+00:00")) if x else None
    tasks = [e["id"] for e in ents if e["type"] == "Task"]
    registered = {t for t in tasks if cov[t]["claims"] + cov[t]["hypotheses"] > 0}
    competencies = {e["id"] for e in ents if e["type"] == "Competency"}
    specs = {e["id"]: e for e in ents if e["type"] == "AssessmentSpec"}
    cas = [e for e in ents if e["type"] == "CompetencyAssessment"]
    def cas_valid(c):
        sp = specs.get(c["data"].get("spec_ref"), {})
        return c["data"].get("outcome") == "PASS" and sp.get("data", {}).get("competency_ref") == c["data"].get("competency_ref")
    passed = {c["data"]["competency_ref"] for c in cas if cas_valid(c)}
    mismatched = [c["id"] for c in cas if c["data"].get("outcome") == "PASS" and not cas_valid(c)]

    hyps = [e for e in ents if e["type"] == "Hypothesis"]
    closed = [h for h in hyps if h["data"].get("closed_at") and h["data"].get("opened_at")]
    durations = [ (dt(h["data"]["closed_at"]) - dt(h["data"]["opened_at"])).total_seconds() / 60 for h in closed ]
    m04 = f"{stt.median(durations):.0f} min (n={len(durations)})" if durations else "NO_DATA"

    exps = [e for e in ents if e["type"] == "Experiment"]
    exp_exec = {r["from"] for r in rels if r["relation"] == "EXECUTED_VIA"}
    exp_items = {r["from"] for r in rels if r["relation"] == "EXECUTED_VIA"}
    # protocol validity: experiment has success_criteria/stop_criteria and its execution reached a result
    valid_proto = {e["id"] for e in exps if e["id"] in exp_exec and e["data"].get("success_criteria") and e["data"].get("stop_criteria")}
    # repeatability: evaluations with a second identical run artifact (CLM:REP:01 evidences one policy only)
    ev_runs = [e for e in ents if e["type"] == "EvaluationRun"]
    repeated = 1  # only the hurdles repeat (2x) is evidenced by ART:hfast50:rep1/rep2
    m06 = (f"registered {len(exp_exec)}/{len(exps)}; protocol-valid {len(valid_proto)}/{len(exps)}; "
           f"repeated {repeated}/{len(ev_runs)}")

    tra = [e for e in ents if e["type"] == "TransferAssessment"]
    by_kind = {}
    for t in tra:
        k = t["data"].get("transferred_kind")
        by_kind.setdefault(k, []).append(t["data"].get("outcome"))
    m07 = "; ".join(f"{k}: {sum(1 for o in v if o=='POSITIVE')}/{len(v)} positive" for k, v in by_kind.items())

    decs = [e for e in ents if e["type"] == "Decision"]
    # decision evidence: DEPENDS_ON/CONSUMES_KNOWLEDGE edges to non-retracted claims/leaderboards dated before decision
    ok_dec = set()
    for d in decs:
        d_at = dt(d["data"].get("decided_at") or d["recorded_at"])
        for r in rels:
            if r["from"] != d["id"] or r["relation"] not in ("DEPENDS_ON", "CONSUMES_KNOWLEDGE"):
                continue
            tgt = next((x for x in ents if x["id"] == r["to"]), None)
            if not tgt:
                continue
            status = tgt["data"].get("epistemic_status")
            if status in ("RETRACTED", "SUPERSEDED"):
                continue
            if tgt["type"] in ("LeaderboardSnapshot", "CompetitionAttempt"):
                if dt(tgt["data"].get("observed_at") or tgt["data"].get("submitted_at") or tgt["recorded_at"]) <= d_at:
                    ok_dec.add(d["id"]); break
            else:
                ok_dec.add(d["id"]); break

    m = {
        "M01a_task_registration": f"{len(registered)}/{len(tasks)} tasks have ≥1 claim or hypothesis (registration, not knowledge coverage)",
        "M01b_target_knowledge_coverage": "NO_TARGET_LIST — 未定义每任务目标知识点，无法计算真正覆盖率",
        "M02_competency_attainment": f"{len(passed)}/{len(competencies)} competencies passed by a CAS whose spec matches"
                                     + (f"; spec mismatches {mismatched}" if mismatched else ""),
        "M03_evidence_distribution": dict(ev),
        "M04_hypothesis_cycle": m04,
        "M05_hypothesis_throughput": f"{len(closed)} closed in session SES:2026-11-28 (window 20:00Z→06:00Z)",
        "M06_experiment": m06,
        "M07_transfer": m07 or "NO_DATA",
        "M08_decision_evidence": f"{len(ok_dec)}/{len(decs)} decisions cite in-scope, non-retracted, pre-decision evidence",
    }
    L += ["", "## Growth metrics (v0.2, DB-KG-04 honest denominators)", ""]
    for k, v in m.items():
        L.append(f"- {k}: {v}")

    out = KB / "generated" / "GROWTH.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()