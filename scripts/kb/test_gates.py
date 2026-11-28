#!/usr/bin/env python3
"""Negative tests: mutate the graph in memory and confirm the validator rejects it."""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_case(name, mutate):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(ROOT / "kb", tmp / "kb")
        shutil.copytree(ROOT / "scripts", tmp / "scripts")
        mutate(tmp / "kb")
        r = subprocess.run([sys.executable, str(tmp / "scripts/kb/validate.py")],
                           capture_output=True, text=True)
        ok = r.returncode != 0
        print(f"{'PASS' if ok else 'FAIL'} {name}: exit={r.returncode}")
        return ok


def edit(path, fn):
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    fn(lines)
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n")


cases = []
cases.append(("claim without evidence", lambda kb: edit(kb / "data/relations.jsonl",
    lambda ls: ls.__setitem__(slice(None), [r for r in ls if r["to"] != "CLM:K13:01"]) and None)
    or edit(kb / "data/knowledge.jsonl",
    lambda ls: [e.pop("source_refs") if e["id"] == "CLM:K13:01" else None for e in ls])))
cases.append(("duplicate id", lambda kb: edit(kb / "data/domain.jsonl",
    lambda ls: ls.append(dict(ls[0])))))
cases.append(("unknown relation target", lambda kb: edit(kb / "data/relations.jsonl",
    lambda ls: ls.append({"from": "CLM:K03:01", "relation": "SUPPORTS", "to": "CLM:does-not-exist"}))))
cases.append(("missing required field", lambda kb: edit(kb / "data/assets.jsonl",
    lambda ls: ls[0]["data"].pop("implementation_ref", None))))

cases.append(("assessment without evidence", lambda kb: edit(kb / "data/relations.jsonl",
    lambda ls: ls.__setitem__(slice(None), [r for r in ls if not (r["from"] == "CAS:reward-audit:hurdles" and r["relation"] == "DEMONSTRATED_BY")]))))
cases.append(("learning activity without performer", lambda kb: edit(kb / "data/relations.jsonl",
    lambda ls: ls.__setitem__(slice(None), [r for r in ls if not (r["from"] == "LA:read-literature" and r["relation"] == "PERFORMED_BY")]))))

ok = all(run_case(n, m) for n, m in cases)
sys.exit(0 if ok else 1)
