"""CPU-tier move lab (research harness, throwaway). See ../README.md."""
from __future__ import annotations

import os
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]  # spikes/move-lab
# override: a lab copied out of the repo (disc-rl kits); a copy right below / (a player's kit at /kit) has no 2 parents
REPO = Path(os.environ.get("DUCKBENCH_REPO") or (LAB.parents[1] if len(LAB.parents) > 1 else LAB.anchor))
# duckref: a copy vendored beside lab/ (a player's RL kit), else the repo's tools/py
sys.path.insert(0, str(LAB if (LAB / "duckref" / "__init__.py").exists() else REPO / "tools" / "py"))

_DEFAULT_HOME = Path.home() / ".duckbench" / "move-lab"  # outside /private/tmp, which macOS may clear
HOME = Path(os.environ.get("MOVE_LAB_HOME", _DEFAULT_HOME))  # runs, checkpoints, lock, hidden-seed secret
# the plant v1 bundle; DUCKBENCH_PLANT_DIR overrides it (a kit outside the repo: the CLI's plant)
PLANT = Path(os.environ.get("DUCKBENCH_PLANT_DIR") or REPO / "packages/sim/plant/microduck-legs-bam-v1")
