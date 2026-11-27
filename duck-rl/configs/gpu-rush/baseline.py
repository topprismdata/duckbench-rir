"""GPU Rush (`gpu-rush`, a 4-duck crowd Challenge): the library move `steer-walk@1` (WARM_MLP; HOST "raw", house medium's
per-node commands in RAW_COMMAND) on the six walk nodes of programs/gpu-rush-1v1-medium.json (the Arena bot house
medium's duckscript: the a_* nodes walk to the nearest GPU, the b_* nodes to the second-nearest when the rival is
clearly closer to the nearest, turning by its bearing; a fall -> stock get_up). Training plays GPU Rush 1v1 (`gpu-rush-1v1`, CHALLENGE) against OPPONENT; the in-loop TS eval scores
the checkpoints in GPU Rush itself (EVAL_CHALLENGE, the Challenge's field of Arena bots, seeds 200-223) and the export
is stamped for `gpu-rush`.
Untrained on check seeds 8-39 in GPU Rush: 0.49 points per round (1st in 11 of 32 rounds).
The reward pays the outcome (+10 a win, -3 a loss or a level time-out, -5 leaving the trained nodes after a fall), +2
per GPU eaten, -2 per GPU the rival eats, closing on the GPU the active node walks to, being upright, smooth actions.
Everything here is yours to change (train.py --help lists the knobs).
"""
import json
from pathlib import Path

import numpy as np

CHALLENGE = "gpu-rush-1v1"  # the training match: one-on-one against OPPONENT
EVAL_CHALLENGE = "gpu-rush"  # the checkpoints scored and the export stamped for the 4-duck crowd Challenge
PROGRAM = "programs/gpu-rush-1v1-medium.json"
TRAINED_NODE = ["a_run", "a_l", "a_r", "b_run", "b_l", "b_r"]  # every walk node (`up` = get_up stays stock)
WARM_START = "alpha_walking"  # the stock graph (export metadata); the weights come from WARM_MLP
WARM_MLP = "steer-walk@1"  # a raw move: speed (vx -0.2..0.65) and wz knobs
HOST = "raw"
RAW_COMMAND = {  # house medium's commands (packages/challenges/baselines/gpu-rush-1v1/house-medium.json)
    "a_run": {"speed": 0.6}, "b_run": {"speed": 0.6},
    "a_l": {"speed": 0.1, "wz": 1.0}, "a_r": {"speed": 0.1, "wz": -1.0},
    "b_l": {"speed": 0.1, "wz": 1.0}, "b_r": {"speed": 0.1, "wz": -1.0},
}
INIT_STD = 0.1
LR = 1e-4
GAMMA = 0.995  # ~4 s ahead at 50 Hz
STAGGER_START = True  # a match is up to 1500 ticks: spread the envs over its seconds

OPPONENT = "house:medium"  # or "house:easy", "stand", a file, "submissions/<label>"; a list; weights
UNBOUND = "end"

# node_i (the active node's index in the duckscript's node order) of the nodes that walk to the second crumb
try:
    _NODES = list(json.loads((Path(__file__).resolve().parents[2] / PROGRAM).read_text())["nodes"])
except OSError:  # the config copied away from its kit: house medium's order
    _NODES = ["a_run", "a_l", "a_r", "b_run", "b_l", "b_r", "up"]
SECOND = [float(_NODES.index(n)) for n in ("b_run", "b_l", "b_r")]

REWARD_PARAMS = dict(w_win=10.0, w_loss=3.0, w_unbound=5.0, w_eat=2.0, w_rival_eat=2.0, w_close=1.0, w_up=0.1,
                     w_smooth=0.05)


def reward(s, p):
    """Per-second rates over the state columns (duck_rl/challenges/gpu_rush.py: Sumo's columns plus the crumbs')."""
    close = np.where(np.isin(s["node_i"], SECOND), s["crumb2_close"], s["crumb_close"])
    return {
        "win": p["w_win"] * s["won"] / 0.02,                       # one-off on the decision row
        "loss": -p["w_loss"] * s["lost"] / 0.02,                   # a loss, or level at 30 s
        "unbound": -p["w_unbound"] * s["unbound"] / 0.02,          # a fall: the duckscript went to get_up
        "eat": p["w_eat"] * s["ate_self"] / 0.02,
        "rival_eat": -p["w_rival_eat"] * s["ate_opp"] / 0.02,
        "close": p["w_close"] * np.clip(close, -0.3, 0.3),         # m/s toward the crumb the node walks to
        "upright": p["w_up"] * s["upright_self"],
        "smooth": -p["w_smooth"] * np.sum((s["action"] - s["prev_action"]) ** 2, axis=1),
    }


TS_EVAL = dict(enabled=True, seeds="200-223", every_s=90.0, nice=10, select="best", final=True)
