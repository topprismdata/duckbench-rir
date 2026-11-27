"""python -m duck_rl.sense <train|refit|check|synth> ... (duck_rl/sense/__init__.py)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def _report(ev: dict) -> None:
    f = ev.pop("_files")
    print(f"wrote {f['json']} (+ .bin), id {f['id']}")
    print(f"  template {ev['template']}, rows train {ev['rows']['train']} / val {ev['rows']['val']} / test {ev['rows']['test']}, "
          f"fit {ev['fit_s']} s ({ev['epochs']} epochs), prep {ev['prep_s']} s, {ev['macs']} MACs / tick (budget 512000)")
    for nm, m in ev["test"].items():
        print(f"  {nm}: " + ", ".join(f"{k} {v}" for k, v in m.items()))
    for g, mm in (ev.get("per_world") or {}).items():
        print(f"  [{g}] " + "; ".join(f"{nm}: " + ", ".join(f"{k} {v}" for k, v in m.items() if k != "n") for nm, m in mm.items()))
    p = ev["parity"]
    print(f"  parity vs the reference evaluator: max |diff| {p['max_abs_diff']} over {p['rows']} held-out rows ({'OK' if p['ok'] else 'FAIL'})")
    for w in ev.get("_warnings") or []:
        print(f"  warning: {w}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m duck_rl.sense", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train a sense on collect data with a label file")
    t.add_argument("--data", required=True, help="collect dir(s), comma-separated")
    t.add_argument("--label", required=True, help="a .py file with label(row)")
    t.add_argument("--template", default="mlp-cam", choices=["mlp-cam", "emb1-head", "conv"])
    t.add_argument("--out", required=True, help="NAME or DIR/NAME: writes NAME.sense.json + NAME.sense.bin")
    t.add_argument("--hidden", help="hidden widths, e.g. 32,32 (the template's by default)")
    t.add_argument("--body", help="body inputs, comma-separated (default: the label file's INPUTS, else the template's)")
    t.add_argument("--cam", help="cam inputs, comma-separated ('' for none)")
    t.add_argument("--no-prev", action="store_true", help="no previous-tick inputs")
    t.add_argument("--split-seed", type=int, default=0)
    t.add_argument("--all-ducks", action="store_true", help="label every duck's rows, not only yours (duck 0)")
    r = sub.add_parser("refit", help="retrain a sense's template, inputs and label on new data")
    r.add_argument("--sense", required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--out", help="write NAME2 instead of overwriting")
    c = sub.add_parser("check", help="validate a sense (the repo's TS check when available) and print its MACs")
    c.add_argument("--sense", required=True)
    c.add_argument("--python", action="store_true", help="the Python mirror only")
    s = sub.add_parser("synth", help="write a synthetic duck-sense-collect/1 dir")
    s.add_argument("--out", required=True)
    s.add_argument("--seeds", default="0-7")
    s.add_argument("--ticks", type=int, default=150)
    s.add_argument("--challenge", default="synth")
    a = ap.parse_args(argv)

    if a.cmd == "synth":
        from .synth import make
        print(make(a.out, _seeds(a.seeds), a.ticks, a.challenge))
        return 0
    if a.cmd == "check":
        from .check import check
        res = check(a.sense, ts=not a.python)
        print(json.dumps(res))
        print(f"{'OK' if res['ok'] else 'PROBLEM: ' + str(res['problem'])}; {res['macs']} MACs per tick of the {res['budget']} budget "
              f"for all of a strategy's senses; id {res['id']} ({res['via']})")
        return 0 if res["ok"] else 1
    from . import train as T
    dirs = [x for x in a.data.split(",") if x]
    if a.cmd == "train":
        src = Path(a.label).read_text()
        inputs = {}
        if a.body is not None:
            inputs["body"] = [x for x in a.body.split(",") if x]
        if a.cam is not None:
            inputs["cam"] = [x for x in a.cam.split(",") if x]
        if a.no_prev:
            inputs["prev"] = False
        hidden = [int(x) for x in a.hidden.split(",")] if a.hidden else None
        ev = T.train(dirs, src, str(Path(a.label)), a.template, a.out, hidden=hidden, inputs=inputs or None, split_seed=a.split_seed,
                     ducks="all" if a.all_ducks else "own")
    else:
        ev = T.refit(a.sense, dirs, a.out)
    ok = ev["parity"]["ok"]
    _report(ev)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
