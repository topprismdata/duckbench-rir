"""Moves for the kit (docs/move-library-design.md §3, §6): the move id, WARM_MLP refs and the lineage of an export.

- The move id is "mv_" + sha256(x.mlp.bin)[:12], the Python mirror of tools/moves.ts moveIdOf (`move_id_of`;
  `python duck_rl/moves.py [x.mlp.json ...]` checks it against `node tools/xp.ts move id`).
- A move ref (WARM_MLP, eval_ts.py --policy) is a path to a duck-mlp/1 x.mlp.json, a move id "mv_<12 hex>" or an
  alias "<slug>@<n>" of the move index. Ids and aliases resolve through the kit's move library (duck_rl/bridge.py
  move_path / move_show): the kit's own repo (`node <REPO>/tools/xp.ts move path`; in an experiment that is the pinned
  repo, so every designer of a duel gets the same pinned library), or the installed CLI's moves pack (standalone).
- `lineage` builds train.lineage of a duck-strategy/1 export: the parent move, the root, this step's and the cumulative
  steps and training time, the recipe hash and the kit version (README "Warm starts").
Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MOVE_ID_RE = re.compile(r"^mv_[0-9a-f]{12}$")
ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}@[1-9][0-9]*$")
STRATEGY_FORMAT = "duck-strategy/1"
LEGACY_FORMATS = ("duck-rl-submission/1", "disc-rl-submission/1")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bin_of(json_path) -> Path:
    """x.mlp.json -> x.mlp.bin."""
    p = Path(json_path)
    if not p.name.endswith(".json"):
        raise ValueError(f"{p}: a duck-mlp/1 policy file ends in .json (its weights: the sibling .bin)")
    return p.with_name(p.name[: -len(".json")] + ".bin")


def move_id_of(path) -> str:
    """"mv_" + sha256(the .bin)[:12]; `path` is an x.mlp.json (its x.mlp.bin beside it) or the .bin itself."""
    p = Path(path)
    return "mv_" + sha256_file(p if p.suffix == ".bin" else bin_of(p))[:12]


def is_library_ref(ref: str) -> bool:
    """A move id or an alias (resolved through the move index), not a path."""
    return bool(MOVE_ID_RE.match(ref) or ALIAS_RE.match(ref))


def xp(repo, *args, timeout: float = 120) -> str:
    """`node <repo>/tools/xp.ts <args>` run in `repo` (DUCKBENCH_REPO = repo): its stdout; RuntimeError on failure.
    tools/xp.ts needs `repo` to be a git work tree (setup.py `git init`s the pinned repo for this). The kit resolves
    moves through duck_rl/bridge.py; this is the monorepo-only id check below."""
    env = dict(os.environ, DUCKBENCH_REPO=str(repo))
    r = subprocess.run(["node", str(Path(repo) / "tools" / "xp.ts"), *map(str, args)], cwd=repo, env=env,
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"tools/xp.ts {' '.join(map(str, args))}: {(r.stderr or r.stdout).strip()[-1500:]}")
    return r.stdout


def resolve_move(ref: str) -> dict:
    """A move ref -> {"id", "json", "bin", "alias"} (alias None when the index has none) by `move path` (bridge). Give a
    path as an absolute path; an id or alias comes from the kit's move index (files materialised if needed)."""
    from duck_rl import bridge  # noqa: PLC0415  (stdlib; lazy so that this file also runs as a script)
    return bridge.move_path(ref)


def resolve_path(json_path) -> dict:
    """An existing x.mlp.json -> the resolve_move dict; the id from the bytes here when the move index cannot be asked
    (then no alias). A .bin the player package does not ship (its moves/, an Arena bot's moves/: kit 1.31.2) comes from
    the CLI, which downloads it on first use: then "bin" is its cached copy."""
    p = Path(json_path).resolve()
    if p.exists() and not bin_of(p).exists():
        try:
            mv = resolve_move(str(p))
        except (RuntimeError, OSError, subprocess.TimeoutExpired, ValueError) as e:
            raise FileNotFoundError(f"{p}: its .bin is missing ({str(e).splitlines()[0][:300]})") from e
        if mv["id"] != move_id_of(mv["bin"]):
            raise RuntimeError(f"{p}: the CLI's copy {mv['bin']} is not {mv['id']}")
        return mv
    if not p.exists() or not bin_of(p).exists():
        raise FileNotFoundError(f"{p} or its .bin is missing")
    try:
        mv = resolve_move(str(p))
    except (RuntimeError, OSError, subprocess.TimeoutExpired, ValueError) as e:
        print(f"[duck-rl] note: no move index lookup for {p.name} ({str(e).splitlines()[0][:200]}); id from the .bin", flush=True)
        mv = {"id": move_id_of(p), "json": str(p), "bin": str(bin_of(p)), "alias": None}
    if mv["id"] != move_id_of(p):
        raise RuntimeError(f"{p}: the move index says {mv['id']}, the Python mirror {move_id_of(p)}")
    return mv


def move_cards(move_id: str) -> list[dict]:
    """The duck-move-card/1 cards of a move (`move show <id>`, bridge, which prints each card as indented JSON); [] for
    a stock move, an unknown id or when the move index cannot be asked."""
    from duck_rl import bridge  # noqa: PLC0415
    try:
        out = bridge.move_show(move_id)
    except (RuntimeError, OSError, subprocess.TimeoutExpired):
        return []
    cards, buf = [], None
    for line in out.splitlines():
        if buf is None and line == "{":
            buf = [line]
        elif buf is not None:
            buf.append(line)
            if line == "}":
                try:
                    c = json.loads("\n".join(buf))
                except ValueError:
                    c = None
                if isinstance(c, dict) and c.get("format") == "duck-move-card/1" and c.get("id") == move_id:
                    cards.append(c)
                buf = None
    return cards


def knobs_of(ref: str, warm: dict | None) -> list[dict]:
    """The knobs a raw command may name (RAW_COMMAND / COMMAND_RANGES keys besides the slot names): the host.knobs of
    the card of `ref` when it is a library ref (an alias, or a move id: the card of the alias it resolved to, warm's
    "alias") and that card is raw; [] for a path (a path names weights, not a card: slot names only), a stock move or
    a walk/stand card. The Python side of tools/moves.ts moveKnobsOf."""
    if warm is None or not is_library_ref(ref) or not warm.get("alias"):
        return []
    card = next((c for c in move_cards(warm["id"]) if c.get("alias") == warm["alias"]), None)
    host = (card or {}).get("host") or {}
    return [k for k in host.get("knobs") or [] if isinstance(k, dict)] if host.get("kind") == "raw" else []


def stock_policy_name(json_path) -> str | None:
    """"alpha_walking" for <repo>/packages/sim/plant/<plant>/policies/alpha_walking.mlp.json (a stock move), else None."""
    p = Path(json_path)
    if p.parent.name == "policies" and p.parent.parent.parent.name == "plant" and p.name.endswith(".mlp.json"):
        return p.name[: -len(".mlp.json")]
    return None


def submission_lineage(json_path, move_id: str) -> dict | None:
    """train.lineage of the submission beside a policy file (submission.json or strategy.json in its dir), when that
    submission binds this move (by pin or by the file), else None."""
    d = Path(json_path).resolve().parent
    for name in ("submission.json", "strategy.json"):
        f = d / name
        if not f.is_file():
            continue
        try:
            s = json.loads(f.read_text())
        except ValueError:
            continue
        binds = [b for b in (s.get("moves") or s.get("policies") or []) if isinstance(b, dict)]
        ref = lambda b: b.get("move") if isinstance(b.get("move"), str) else b.get("mlp")  # noqa: E731
        hit = any(b.get("pin") == move_id or (isinstance(ref(b), str) and (d / ref(b)).resolve() == Path(json_path).resolve())
                  for b in binds)
        lin = (s.get("train") or {}).get("lineage") if isinstance(s.get("train"), dict) else None
        if hit and isinstance(lin, dict):
            return lin
    return None


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def parent_of(warm_start: str, warm: dict | None = None, init: dict | None = None, cards: bool = True) -> dict:
    """Where the trained weights start, for train.lineage: {parent, parent_alias?, parent_ckpt?, root, base} where
    base = the parent's cumulative {steps, train_s} (None where unknown; 0 for a stock parent: compute inherited from the
    stock skills or the library is not charged) and `notes` say what is unknown. Steps count what the weights were
    trained on; train_s counts what producing them cost (whole runs, selection included).
      init: {"id", "path", "steps" (the checkpoint's), "lineage" (the train lineage of the run or submission the
            checkpoint comes from, or None)}: --init (continue from a checkpoint: its weights are the parent)
      warm: the resolved WARM_MLP (resolve_move / resolve_path); its card's lineage is looked up unless cards=False
      neither: the stock policy WARM_START."""
    notes: list[str] = []
    if init is not None:
        out = dict(parent=init["id"], parent_ckpt=init["path"])
        lin = init.get("lineage")
        if isinstance(lin, dict):
            out["root"] = lin.get("root") or f"stock:{warm_start}"
            cum = lin.get("cumulative") if isinstance(lin.get("cumulative"), dict) else {}
            this = lin.get("this_step") if isinstance(lin.get("this_step"), dict) else {}
            # steps: that run's parent's cumulative + the checkpoint's own steps; train_s: that whole run's cost
            c, t, o = _num(cum.get("steps")), _num(this.get("steps")), _num(init.get("steps"))
            out["base"] = {"steps": None if c is None or t is None or o is None else c - t + o,
                           "train_s": _num(cum.get("train_s"))}
        else:
            out["root"] = f"stock:{warm_start}"
            out["base"] = {"steps": None, "train_s": None}
            notes.append(f"--init {Path(init['path']).name}: no train lineage beside it (root assumed stock:{warm_start})")
    elif warm is not None:
        out = dict(parent=warm["id"])
        if warm.get("alias"):
            out["parent_alias"] = warm["alias"]
        stock = stock_policy_name(warm["json"])
        card_lin = next((c["lineage"] for c in (move_cards(warm["id"]) if cards else [])
                         if isinstance(c.get("lineage"), dict) and c["lineage"].get("root")), None)
        sub_lin = None if card_lin is not None else submission_lineage(warm["json"], warm["id"])
        lin = card_lin or sub_lin
        if lin is not None:
            out["root"] = lin["root"]
            cum = lin.get("cumulative") if isinstance(lin.get("cumulative"), dict) else {}
            out["base"] = {k: _num(cum.get(k)) for k in ("steps", "train_s")}
            src = "card" if card_lin is not None else "submission"
            if None in out["base"].values():
                notes.append(f"the parent's {src} lineage lacks cumulative {', '.join(k for k, v in out['base'].items() if v is None)}")
        elif stock is not None:
            out["root"] = f"stock:{stock}"
            out["base"] = {"steps": 0, "train_s": 0}
        else:
            out["root"] = f"stock:{warm_start}"
            out["base"] = {"steps": None, "train_s": None}
            notes.append(f"the parent {warm['id']} has no card and no submission lineage (root assumed stock:{warm_start})")
    else:
        out = dict(parent=f"stock:{warm_start}", root=f"stock:{warm_start}", base={"steps": 0, "train_s": 0})
    out["notes"] = notes
    return out


def lineage(parent: dict, this_step: dict, recipe_sha256: str, kit: str) -> dict:
    """train.lineage: {parent, parent_alias?, parent_ckpt?, root, this_step, cumulative, recipe, kit, notes?}.
    cumulative = the parent's cumulative + this step, per key; a key whose parent value is unknown is null (both
    unknown: cumulative null) and `cumulative_note` says why."""
    lin = {k: parent[k] for k in ("parent", "parent_alias", "parent_ckpt", "root") if parent.get(k) is not None}
    lin["this_step"] = this_step
    base = parent.get("base") or {}
    cum = {}
    for k in ("steps", "train_s"):
        b, t = _num(base.get(k)), _num(this_step.get(k))
        cum[k] = None if b is None or t is None else (round(b + t, 1) if k == "train_s" else int(b + t))
    unknown = [k for k, v in cum.items() if v is None]
    lin["cumulative"] = None if len(unknown) == len(cum) else cum
    if unknown:
        lin["cumulative_note"] = f"the parent's cumulative {' and '.join(unknown)} unknown" + (" (notes)" if parent.get("notes") else "")
    lin["recipe"] = f"sha256:{recipe_sha256}"
    lin["kit"] = kit
    if parent.get("notes"):
        lin["notes"] = parent["notes"]
    return lin


def _check(paths: list[str]) -> int:
    """move_id_of vs `node tools/xp.ts move id` on each file (default: the stock alpha_walking policy)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import duck_rl  # noqa: PLC0415  (REPO; it imports the move lab)
    if duck_rl.REPO is None:
        raise SystemExit("the move id check runs in the monorepo (tools/xp.ts move id)")
    files = [Path(p).resolve() for p in paths] or [duck_rl.PLANT / "policies" / "alpha_walking.mlp.json"]
    bad = 0
    for f in files:
        py, ts = move_id_of(f), xp(duck_rl.REPO, "move", "id", str(f)).strip()
        bad += py != ts
        print(f"{'ok  ' if py == ts else 'DIFF'} {py} (python) {ts} (tools/xp.ts move id) {f}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_check(sys.argv[1:]))
