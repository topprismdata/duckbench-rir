#!/usr/bin/env python3
"""Small query CLI over kb/generated/graph.sqlite.

  query.py support <id>    what SUPPORTS/REFUTES this id
  query.py open            hypotheses still OPEN
  query.py exclusions      active exclusions and reopen conditions
  query.py chain <id>      DERIVES_FROM chain up to 5 hops
"""
import json, sqlite3, sys
from pathlib import Path

DB = Path(__file__).resolve().parents[2] / "kb/generated/graph.sqlite"


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    con = sqlite3.connect(DB)
    cmd = sys.argv[1]
    if cmd == "support" and len(sys.argv) > 2:
        i = sys.argv[2]
        for src, rel in con.execute("SELECT src, rel FROM relations WHERE dst=? AND rel IN ('SUPPORTS','REFUTES','EQUIVALENT_TO')", (i,)):
            row = con.execute("SELECT type,label FROM entities WHERE id=?", (src,)).fetchone()
            print(f"{rel:9s} {src}  {row[0] if row else '?'}: {row[1] if row else ''}")
    elif cmd == "open":
        for i, label, data in con.execute("SELECT id,label,data FROM entities WHERE type='Hypothesis'"):
            d = json.loads(data)
            if d.get("status") == "OPEN":
                print(f"{i}  {d['statement']}  (falsifier: {d.get('falsifier')})")
    elif cmd == "exclusions":
        for i, data in con.execute("SELECT id,data FROM entities WHERE type='Exclusion'"):
            d = json.loads(data)
            print(f"{i}: {d['route']} — reopen if {d['reopen_condition']}")
    elif cmd == "chain" and len(sys.argv) > 2:
        cur, seen = sys.argv[2], set()
        for _ in range(5):
            nxt = con.execute("SELECT dst FROM relations WHERE src=? AND rel='DERIVES_FROM'", (cur,)).fetchone()
            if not nxt or nxt[0] in seen: break
            seen.add(nxt[0]); print(f"{cur} ← {nxt[0]}"); cur = nxt[0]
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
