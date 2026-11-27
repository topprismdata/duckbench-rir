"""duck-rl: PPO on any DuckBench Challenge with the move lab (rsl_rl + MLX learner + numba BAM env). The unified RL kit,
built from the disc-rl kit v2 (Lazy Susan) and the wheel-rl kit (Hamster wheel); spikes/duck-rl/README.md.

Paths (all overridable by env, so a kit copy works outside the repo):
  KIT   the directory that holds duck_rl/ (the repo's spikes/duck-rl/kit, a designer's copy, or a player's project)
  REPO  the TS sim, the Challenges, the plant, the eval (read-only; duck_rl/bridge.py): DUCKBENCH_REPO, else KIT/REPO_PATH,
        else the repo above KIT. None in standalone mode (KIT holds duckbench-kit.json: the installed CLI is the TS
        side, and it names the plant)
  LAB   KIT/lab if the kit has its own copy of the move lab (designer copies and players' kits do), else
        REPO/spikes/move-lab; a kit that vendors duckref/ beside lab/ uses that copy (lab/__init__.py)
  HOME  DUCK_RL_HOME, else KIT/home: scenes, banks, runs (checkpoints), evals, caches
Importing this package sets the move lab's home and plant before `import lab`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import bridge

KIT_VERSION = "1.31.2"
if bridge.STANDALONE and bridge.CONFIG.get("kit_version") != KIT_VERSION:
    print(f"[duck-rl] warning: {bridge.KIT_JSON.name} is for kit {bridge.CONFIG.get('kit_version')}, this folder is kit "
          f"{KIT_VERSION}: run `duckbench kit init . --force` in this folder to bring the CLI's kit in (budget.json and "
          f"its ledger stay; kit files you changed are replaced)", file=sys.stderr, flush=True)

KIT = bridge.KIT
REPO = bridge.REPO
HOME = Path(os.environ.get("DUCK_RL_HOME") or KIT / "home").expanduser()
if (KIT / "lab" / "__init__.py").exists():
    LAB_DIR = KIT
elif bridge.STANDALONE:
    raise SystemExit(f"{KIT}: a standalone kit (duckbench-kit.json) needs its own lab/ (duckbench kit init copies it)")
else:
    LAB_DIR = REPO / "spikes" / "move-lab"
if bridge.STANDALONE:
    os.environ["DUCKBENCH_PLANT_DIR"] = str(bridge.plant_dir())  # lab.PLANT and duckref's plant v1 (env workers too)
else:
    os.environ["DUCKBENCH_REPO"] = str(REPO)  # the lab (and spawned env workers) resolve the plant from here
    os.environ.pop("DUCKBENCH_PLANT_DIR", None)  # an inherited one (a player's kit shell) would override the repo's plant
os.environ["MOVE_LAB_HOME"] = str(HOME)
for p in (str(KIT), str(LAB_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import lab  # noqa: E402

assert Path(lab.HOME) == HOME, (lab.HOME, HOME)
PLANT = lab.PLANT
SCENES = HOME / "scenes"
RUNS = HOME / "runs"
BANKS = HOME / "banks"
TS = REPO / "spikes" / "duck-rl" / "ts" if REPO is not None else None  # monorepo only; the kit calls bridge.run_ts
SCENE_SOURCES = bridge.SCENE_SOURCES


def scene_paths(challenge: str, variant: str | None = None) -> tuple[Path, Path]:
    """HOME/scenes/<challenge>.xml (the world, absolute meshdir) and <challenge>.json (the info: ts/prepare.ts); a spike Challenge's
    variant ("diameter=1.9,mass=5") gets <challenge>@diameter=1.9,mass=5.xml/.json."""
    stem = challenge if not variant else f"{challenge}@{variant.replace(' ', '')}"
    return SCENES / f"{stem}.xml", SCENES / f"{stem}.json"


def ensure_scene(challenge: str, force: bool = False, variant: str | None = None) -> tuple[Path, Path]:
    """Export the Challenge's world + info from the TS side (prepare --challenge --scene --info [--variant]), again when a source
    is newer than the export (bridge.scene_stamp) or the wasm-compiled world <stem>.mjb is missing (kit 1.16.0: env.py
    world_model loads it; scenes exported by an older kit have none). Returns (scene xml, info json)."""
    xml, info = scene_paths(challenge, variant)
    mjb, cam = xml.with_suffix(".mjb"), xml.with_suffix(".camera.mjb")  # cam: the palette camera move's world (env.py)
    if xml.exists() and info.exists() and mjb.exists() and cam.exists() and not force and info.stat().st_mtime >= bridge.scene_stamp():
        return xml, info
    SCENES.mkdir(parents=True, exist_ok=True)
    bridge.run_ts("prepare", ["--challenge", challenge, "--scene", xml, "--info", info, *(["--variant", variant] if variant else [])],
                  check=True)
    return xml, info


def load_info(challenge: str) -> dict:
    """The Challenge info (exported first if missing or stale)."""
    return json.loads(ensure_scene(challenge)[1].read_text())
