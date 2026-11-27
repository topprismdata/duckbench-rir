"""The kit's link to the TS side (the sim, the Challenges, the eval) and to the move library: one module, two modes.

  standalone  the kit dir holds duckbench-kit.json (written by `duckbench kit init`, docs/player-kit-decisions-log.md
              PK-04): every TS call runs the installed CLI, cli + ["kit", sub, ...]; the plant is its plant_dir; move
              refs resolve through the CLI (its moves pack). There is no repo: REPO is None.
  monorepo    no duckbench-kit.json: `node <REPO>/spikes/duck-rl/ts/<sub>.ts` and `node <REPO>/tools/xp.ts move
              path|show`, as before kit 1.3.0. REPO = DUCKBENCH_REPO, else KIT/REPO_PATH (one line, written by
              setup.py), else the repo above the kit; it must hold spikes/duck-rl/ts/eval.ts.

The TS subcommands take the same arguments and print the same output in both modes:
  prepare    ts/prepare.ts     the Challenge's world XML (+ its wasm-compiled MJB) + info JSON; a duckscript validated and compiled
  eval       ts/eval.ts        the TS eval (submissions, programs + moves, seeds, banks, trajectories)
  check-mlp  ts/check_mlp.ts   a duck-mlp/1 through the TS runners
Importing it checks the link (a missing CLI, plant or repo exits with what to run) and does not import the move lab.
Stdlib only.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
# A kit folder made before the DuckBench rename holds duck-arena-kit.json (LEGACY_KIT_JSON): read it when the new one is missing.
LEGACY_KIT_JSON = KIT / "duck-arena-kit.json"
KIT_JSON = KIT / "duckbench-kit.json" if (KIT / "duckbench-kit.json").exists() or not LEGACY_KIT_JSON.exists() else LEGACY_KIT_JSON
RELINK = "run `duckbench kit link .` in this folder (or `duckbench kit init <dir>` for a fresh copy)"
CONFIG: dict | None = json.loads(KIT_JSON.read_text()) if KIT_JSON.exists() else None
STANDALONE = CONFIG is not None
MODE = "standalone" if STANDALONE else "monorepo"
if STANDALONE:
    REPO = None
    CLI = [str(c) for c in CONFIG["cli"]]  # [node, <pkg>/dist/main.js]
    for _f in CLI[1:]:
        if not Path(_f).exists():
            raise SystemExit(f"{KIT_JSON.name} names a duckbench CLI that is gone ({_f}): {RELINK}")
    if not (Path(CLI[0]).exists() or shutil.which(CLI[0])):
        _node = shutil.which("node")
        if _node is None:
            raise SystemExit(f"{KIT_JSON.name} names a Node that is gone ({CLI[0]}) and there is no `node` on PATH: {RELINK}")
        print(f"[duck-rl] warning: {KIT_JSON.name} names a Node that is gone ({CLI[0]}): using {_node}; {RELINK}",
              file=sys.stderr, flush=True)
        CLI[0] = _node
    if not (Path(CONFIG["plant_dir"]).expanduser() / "manifest.json").exists():
        raise SystemExit(f"{KIT_JSON.name} names a plant that is gone ({CONFIG['plant_dir']}): {RELINK}")
else:
    _rp = KIT / "REPO_PATH"
    _up = KIT.parents[2] if len(KIT.parents) > 2 else KIT.anchor  # the repo above spikes/duck-rl/kit
    REPO = Path(os.environ.get("DUCKBENCH_REPO") or (_rp.read_text().strip() if _rp.exists() else _up)).expanduser()
    if not (REPO / "spikes" / "duck-rl" / "ts" / "eval.ts").exists():
        raise SystemExit(f"this kit isn't linked to a duckbench CLI (no {KIT_JSON.name} in {KIT}, and {REPO} is not a "
                         f"duckbench checkout): {RELINK}")
TS_SCRIPTS = {"prepare": "prepare.ts", "eval": "eval.ts", "check-mlp": "check_mlp.ts"}
# What an exported scene/info depends on in the monorepo: the Challenges, the props, the metrics, the schema, the exporter.
# Kit 1.16.0: the world's MJB also depends on the sim's compile (plant.ts) and its MuJoCo (packages/sim/package.json).
SCENE_SOURCES = ("packages/challenges/src", "packages/schema/src/challenge.ts", "packages/sim/src/runtime/metrics.ts",
                 "packages/sim/src/plant.ts", "packages/schema/src/palette.ts", "packages/sim/package.json",
                 "spikes/duck-rl/ts/prepare.ts", "spikes/duck-rl/ts/spike_challenges.ts")


def ts_argv(sub: str) -> list[str]:
    """The command that runs TS subcommand `sub` (prepare | eval | check-mlp); append its arguments."""
    if sub not in TS_SCRIPTS:
        raise ValueError(f"TS subcommand {sub!r}: one of {', '.join(TS_SCRIPTS)}")
    if STANDALONE:
        return [*CLI, "kit", sub]
    return ["node", str(REPO / "spikes" / "duck-rl" / "ts" / TS_SCRIPTS[sub])]


def cwd() -> Path:
    """Where TS commands run: the repo (monorepo) or the kit dir (standalone). Pass absolute paths either way."""
    return KIT if STANDALONE else REPO


def run_ts(sub: str, args, **kw) -> subprocess.CompletedProcess:
    """subprocess.run(ts_argv(sub) + args) in cwd(); keyword arguments pass through (check, capture_output, ...)."""
    return subprocess.run([*ts_argv(sub), *map(str, args)], cwd=cwd(), **kw)


def _move(verb: str, ref: str, timeout: float = 120) -> str:
    """`xp move <verb> <ref>` (monorepo: node tools/xp.ts, which needs REPO to be a git work tree; standalone: the CLI's
    kit move-<verb>): its stdout; RuntimeError on failure."""
    if STANDALONE:
        cmd, env = [*CLI, "kit", f"move-{verb}", ref], None
    else:
        cmd, env = ["node", str(REPO / "tools" / "xp.ts"), "move", verb, ref], dict(os.environ, DUCKBENCH_REPO=str(REPO))
    r = subprocess.run(cmd, cwd=cwd(), env=env, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"move {verb} {ref}: {(r.stderr or r.stdout).strip()[-1500:]}")
    return r.stdout


def move_path(ref: str) -> dict:
    """A move ref (an absolute x.mlp.json path, a move id mv_<12 hex> or an alias <slug>@<n>) -> {"id", "json", "bin",
    "alias"} (alias None when the index has none); files are materialised if needed."""
    return json.loads(_move("path", ref).strip().splitlines()[-1])


def move_show(ref: str) -> str:
    """The move's cards as `xp move show` prints them (each duck-move-card/1 as indented JSON)."""
    return _move("show", ref)


def post_done(session: str, runs: Path | str, timeout: float = 15.0, crashed: str | None = None) -> str | None:
    """Kit 1.16.1 (LB-207): the training ended; `duckbench kit done` posts "Training done" (start -> now and the exported
    weights' run) to the human's page on the site, when the CLI is logged in; `crashed` (kit 1.26.0: the error, one line)
    posts "Training crashed: <error>" instead. Either way the CLI queues the training's files for upload. Never raises:
    its one output line, else None (not logged in, nothing to say, or it failed / took longer than `timeout`)."""
    cli = [*CLI] if STANDALONE else ["node", str(REPO / "apps" / "cli" / "src" / "main.ts")]
    extra = [f"--crashed={crashed}"] if crashed is not None else []  # (= form: an error may start with "-")
    try:
        r = subprocess.run([*cli, "kit", "done", "--training", session, "--runs-dir", str(runs), *extra], cwd=cwd(),
                           capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    out = (r.stdout or "").strip().splitlines()
    return out[-1] if out else None


_CAMERA_WORLDS: dict[tuple[str, str], str | None] = {}


def camera_world(challenge_id: str, picture: str, timeout: float = 60.0) -> str | None:
    """Kit 1.28.0: the Challenge's current camera world for `picture` ("sha256:<64 hex>", packages/sim
    runtime/camera-world.ts; what `duckbench sense collect` writes in meta.json and a camera move card's row carries):
    tools/camera-world.ts (monorepo), or the CLI bundle run with DUCK_ARENA_CAMERA_WORLD (standalone, as its card lint
    does). Cached per process; None when it can't be computed (an unknown Challenge, no node)."""
    key = (challenge_id, picture)
    if key not in _CAMERA_WORLDS:
        if STANDALONE:
            cmd, env = [*CLI], dict(os.environ, DUCK_ARENA_CAMERA_WORLD=f"{challenge_id}\n{picture}")
        else:
            cmd, env = ["node", str(REPO / "tools" / "camera-world.ts"), challenge_id, picture], None
        try:
            r = subprocess.run(cmd, cwd=cwd(), env=env, capture_output=True, text=True, timeout=timeout)
            h = (r.stdout or "").strip().splitlines()[-1:] if r.returncode == 0 else []
            _CAMERA_WORLDS[key] = h[0] if h and h[0].startswith("sha256:") and len(h[0]) == 71 else None
        except (OSError, subprocess.SubprocessError):
            _CAMERA_WORLDS[key] = None
    return _CAMERA_WORLDS[key]


def plant_dir() -> Path:
    """The plant v1 bundle (manifest.json, plant.xml, policies/)."""
    if STANDALONE:
        return Path(CONFIG["plant_dir"]).expanduser()
    return REPO / "packages" / "sim" / "plant" / "microduck-legs-bam-v1"


def scene_stamp() -> float:
    """The newest mtime of what an exported scene/info depends on: the TS sources (monorepo), or duckbench-kit.json
    and the CLI it names (standalone: a new CLI or `kit init` re-exports the scenes)."""
    if STANDALONE:
        files = [KIT_JSON, *(Path(c) for c in CLI[1:])]
    else:
        files = []
        for s in SCENE_SOURCES:
            p = REPO / s
            files += list(p.rglob("*.ts")) if p.is_dir() else [p]
    return max((f.stat().st_mtime for f in files if f.exists()), default=0.0)
