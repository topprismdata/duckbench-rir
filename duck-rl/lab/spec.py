"""Load and statically check a move's move.py (the only file an author edits)."""
from __future__ import annotations

import ast
import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import PLANT
from .duck import SLOTS

WARM_STARTS = ["alpha_walking", "velstand", "alpha_stand", "alpha_sitstand", "alpha_ground_pick", "roulade",
               "ball_kick_left", "ball_kick_right"]
TERMINATION_CATALOGUE = {"max_tilt_deg": (5.0, 45.0), "min_height": (0.07, 0.12), "max_disp_m": (0.05, 20.0),
                         "max_heading_drift_deg": (5.0, 180.0)}
MAX_TERMS, MAX_LINES = 8, 150
BANNED_CALLS = {"open", "eval", "exec", "compile", "getattr", "setattr", "delattr", "__import__", "input", "globals",
                "locals", "vars", "breakpoint", "exit", "quit", "print"}
BANNED_NAMES = {"os", "sys", "subprocess", "pathlib", "random", "time", "importlib", "builtins"}
Z_WARN, Z_FAIL = 3.0, 6.0  # |command - normaliser mean| / normaliser std of the warm-start skill


@dataclass
class Move:
    path: Path
    slug: str
    warm_start: str
    lo: np.ndarray
    hi: np.ndarray
    eval_cmd: np.ndarray
    episode_s: float
    terminations: dict
    terms: dict
    curriculum: list
    reward_terms: object
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def weights(self, frac: float) -> dict:
        """Curriculum stage weights at this fraction of the training budget (step function, default 1)."""
        w = {k: 1.0 for k in self.terms}
        for start, stage in self.curriculum:
            if frac >= start:
                w.update(stage)
        return w

    def stage(self, frac: float) -> int:
        return sum(1 for s, _ in self.curriculum if frac >= s)

    def env_cfg(self) -> dict:
        return dict(policy=self.warm_start, lo=self.lo.tolist(), hi=self.hi.tolist(), episode_s=self.episode_s,
                    terminations=self.terminations)


def ast_check(src: str) -> list[str]:
    errs = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    n_lines = len(src.splitlines())
    if n_lines > MAX_LINES:
        errs.append(f"{n_lines} lines > {MAX_LINES}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ("numpy", "math"):
                    errs.append(f"line {node.lineno}: import {a.name} (only numpy and math)")
        elif isinstance(node, ast.ImportFrom):
            errs.append(f"line {node.lineno}: 'from ... import' is not allowed (use import numpy as np / import math)")
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef, ast.With, ast.Try,
                               ast.While, ast.Yield, ast.YieldFrom, ast.Await)):
            errs.append(f"line {node.lineno}: {type(node).__name__} is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            errs.append(f"line {node.lineno}: call to {node.func.id}() is not allowed")
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in ("random", "load", "save",
                                                                                             "fromfile", "tofile", "memmap", "loadtxt", "savetxt", "genfromtxt")):
            errs.append(f"line {node.lineno}: attribute .{node.attr} is not allowed")
        elif isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            errs.append(f"line {node.lineno}: name {node.id} is not allowed")
    for node in tree.body:  # top level: docstring, imports, constant assignments, function defs
        ok = isinstance(node, (ast.Import, ast.FunctionDef, ast.Assign, ast.AnnAssign)) or (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
        if not ok:
            errs.append(f"line {node.lineno}: top-level {type(node).__name__} is not allowed")
    return errs


def _slot_vec(d: dict, what: str, errs: list) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.zeros(13), np.zeros(13)
    for k, v in (d or {}).items():
        if k not in SLOTS:
            errs.append(f"{what}: unknown slot {k!r} (slots: {', '.join(SLOTS)})")
            continue
        i = SLOTS.index(k)
        a, b = (v, v) if isinstance(v, (int, float)) else tuple(v)
        if a > b:
            errs.append(f"{what}[{k}]: lo {a} > hi {b}")
        lo[i], hi[i] = a, b
    return lo, hi


def load(move_dir: str | Path) -> Move:
    move_dir = Path(move_dir).resolve()
    f = move_dir / "move.py"
    src = f.read_text()
    errs = ast_check(src)
    if errs:
        return Move(f, move_dir.name, "", np.zeros(13), np.zeros(13), np.zeros(13), 10.0, {}, {}, [], None, errs)
    spec = importlib.util.spec_from_file_location(f"move_{move_dir.name}", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    warns = []
    ws = getattr(mod, "WARM_START", None)
    if ws not in WARM_STARTS:
        errs.append(f"WARM_START {ws!r} not in {WARM_STARTS}")
    lo, hi = _slot_vec(getattr(mod, "COMMAND", {}), "COMMAND", errs)
    ev, _ = _slot_vec(getattr(mod, "EVAL_COMMAND", {}), "EVAL_COMMAND", errs)
    if np.any(ev < lo - 1e-9) or np.any(ev > hi + 1e-9):
        errs.append("EVAL_COMMAND must lie inside the COMMAND training ranges")
    ep = float(getattr(mod, "EPISODE_S", 10.0))
    if not 2.0 <= ep <= 20.0:
        errs.append(f"EPISODE_S {ep} outside [2, 20]")
    terms = dict(getattr(mod, "TERMS", {}))
    if not 1 <= len(terms) <= MAX_TERMS:
        errs.append(f"TERMS has {len(terms)} entries (1..{MAX_TERMS})")
    for k, v in terms.items():
        if v not in ("bonus", "penalty"):
            errs.append(f"TERMS[{k!r}] = {v!r} (must be 'bonus' or 'penalty')")
    term_rules = dict(getattr(mod, "TERMINATIONS", {}) or {})
    for k, v in term_rules.items():
        if k not in TERMINATION_CATALOGUE:
            errs.append(f"TERMINATIONS: unknown {k!r} (catalogue: {list(TERMINATION_CATALOGUE)})")
        elif not TERMINATION_CATALOGUE[k][0] <= v <= TERMINATION_CATALOGUE[k][1]:
            errs.append(f"TERMINATIONS[{k}] = {v} outside {TERMINATION_CATALOGUE[k]}")
    cur = list(getattr(mod, "CURRICULUM", []) or [])
    for st in cur:
        if not (isinstance(st, (tuple, list)) and len(st) == 2 and 0.0 <= st[0] < 1.0 and isinstance(st[1], dict)):
            errs.append(f"CURRICULUM stage {st!r}: must be (start_fraction in [0,1), {{term: weight >= 0}})")
            continue
        for k, w in st[1].items():
            if k not in terms or not (isinstance(w, (int, float)) and w >= 0):
                errs.append(f"CURRICULUM weight {k!r}={w!r}: unknown term or negative weight")
    if [s for s, _ in cur if isinstance(s, (int, float))] != sorted(s for s, _ in cur if isinstance(s, (int, float))):
        errs.append("CURRICULUM stages must be sorted by start fraction")
    fn = getattr(mod, "reward_terms", None)
    if not callable(fn):
        errs.append("reward_terms(duck) is missing")
    if ws in WARM_STARTS:  # command vs the warm-start skill's normaliser: out-of-distribution commands
        from duckref.mlp import DuckMlp
        ref = DuckMlp(PLANT / "policies" / f"{ws}.mlp.json")
        names = json.loads((PLANT / "manifest.json").read_text())["policies"][ws]["command_names"]
        live = list(range(3)) + (list(range(3, 7)) if "head_pose" in names else []) + (
            list(range(7, 13)) if "body_pose" in names else [])
        for i in range(13):
            if i not in live and (lo[i] != 0 or hi[i] != 0):
                warns.append(f"slot {SLOTS[i]} is not in {ws}'s obs (always 0): its range has no effect")
                continue
            if i not in live:
                continue
            z = max(abs(lo[i] - ref.sub[48 + i]), abs(hi[i] - ref.sub[48 + i])) / ref.div[48 + i]
            msg = (f"slot {SLOTS[i]} range [{lo[i]}, {hi[i]}] is {z:.1f} std from {ws}'s training mean "
                   f"{ref.sub[48 + i]:.3f} (std {ref.div[48 + i]:.3f})")
            if z > Z_FAIL:
                errs.append(msg + f" > {Z_FAIL}: far out of distribution")
            elif z > Z_WARN:
                warns.append(msg)
    return Move(f, move_dir.name, ws or "", lo, hi, ev, ep, term_rules, terms, cur, fn, errs, warns)


def audit_terms(move: Move, duck: dict, where: str) -> tuple[dict, list]:
    """Run reward_terms on a batch; check keys, shapes, finiteness, signs. Returns (terms, errors)."""
    errs = []
    n = duck["height"].shape[0]
    try:
        out = move.reward_terms(duck)
    except Exception as e:  # noqa: BLE001
        return {}, [f"{where}: reward_terms raised {type(e).__name__}: {e}"]
    if not isinstance(out, dict) or set(out) != set(move.terms):
        return {}, [f"{where}: returned keys {sorted(out) if isinstance(out, dict) else type(out)} != TERMS {sorted(move.terms)}"]
    res = {}
    for k, v in out.items():
        v = np.broadcast_to(np.asarray(v, np.float64), (n,)) if np.ndim(v) == 0 else np.asarray(v, np.float64)
        if v.shape != (n,):
            errs.append(f"{where}: term {k} has shape {v.shape}, expected ({n},)")
            continue
        if not np.all(np.isfinite(v)):
            errs.append(f"{where}: term {k} has non-finite values")
        if move.terms[k] == "bonus" and v.min() < -1e-9:
            errs.append(f"{where}: bonus {k} is negative (min {v.min():.3g})")
        if move.terms[k] == "penalty" and v.max() > 1e-9:
            errs.append(f"{where}: penalty {k} is positive (max {v.max():.3g}): the policy would farm it")
        res[k] = v
    if res:
        tot = np.abs(sum(res.values()))
        if tot.max() > 50:
            errs.append(f"{where}: |sum of terms| up to {tot.max():.3g} > 50 per step (keep terms O(1))")
    return res, errs
