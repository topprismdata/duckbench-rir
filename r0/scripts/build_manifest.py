#!/usr/bin/env python3
"""Build r0/manifest.json: sha256 of every R0 deliverable and key source."""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["r0", "r1", "duckbench-research-handoff"]
KEY_SOURCES = [
    "duckbench-0.3.87-5a1437de.tgz",
    "op3_soccer.zip",
    "soccer-deepmind/soccer-paper-arxiv-2304.13653v2.pdf",
]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk(base: Path):
    for p in sorted(base.rglob("*")):
        if p.is_file() and not p.name.startswith(".") and "__pycache__" not in p.parts:
            yield p


def main():
    out = {}
    for t in TARGETS:
        base = ROOT / t
        for p in walk(base):
            out[str(p.relative_to(ROOT))] = sha(p)
    for name in KEY_SOURCES:
        p = ROOT / "sources" / name
        if p.exists():
            out[str(p.relative_to(ROOT))] = sha(p)
    (ROOT / "r0" / "manifest.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print(f"{len(out)} files hashed -> r0/manifest.json")


if __name__ == "__main__":
    main()