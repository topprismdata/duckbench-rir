"""Wobbly Raft (`wobbly-raft`, a duel), from the stock walk: the stock walk (alpha_walking) on the four walk and stand
nodes of programs/wobbly-raft-medium.json (the Arena bot house medium's duckscript: `hold` stands; past 25 cm from the
raft's centre `seek` / `left` / `right` walk back to it; a fall -> stock get_up); walk-stand host. Untrained, it plays
as house medium.
Untrained on check seeds 8-39 x 2 sides against OPPONENT house:medium: 0.50 points per round (32 W 32 L of 64).
Training plays OPPONENT from either side; an episode ends at the decision or when the duckscript leaves the trained
nodes after a fall. The reward pays the outcome (+10 a win, -3 a loss or both aboard at 30 s, -5 leaving the trained
nodes), staying near the centre, the rival sliding outward, being upright, smooth actions.
The in-loop TS eval plays house medium on seeds 200-215, both sides.
Everything here is yours to change (train.py --help lists the knobs).
"""
import numpy as np

CHALLENGE = "wobbly-raft"
PROGRAM = "programs/wobbly-raft-medium.json"
TRAINED_NODE = ["hold", "seek", "left", "right"]  # every walk and stand node (`up` = get_up stays stock)
WARM_START = "alpha_walking"  # the stock walking policy: the untrained start = house medium
INIT_STD = 0.1
LR = 1e-4
GAMMA = 0.995  # ~4 s ahead at 50 Hz
STAGGER_START = True  # a match is up to 1500 ticks: spread the envs over its seconds

OPPONENT = "house:medium"  # or "house:easy", "house:hard", "stand", a file, "submissions/<label>"; a list; weights
EVAL_OPPONENTS = None  # None: OPPONENT's
UNBOUND = "end"

REWARD_PARAMS = dict(w_win=10.0, w_loss=3.0, w_unbound=5.0, w_centre=0.5, w_attack=1.0, w_up=0.1, w_smooth=0.05)


def reward(s, p):
    """Per-second rates over the state columns (duck_rl/challenges/wobbly_raft.py: Sumo's columns plus raft_tilt)."""
    return {
        "win": p["w_win"] * s["won"] / 0.02,                       # one-off on the decision row
        "loss": -p["w_loss"] * s["lost"] / 0.02,                   # a loss, or both aboard at 30 s
        "unbound": -p["w_unbound"] * s["unbound"] / 0.02,          # a fall: the duckscript went to get_up
        "centre": -p["w_centre"] * s["d_self"],                    # m from the raft's centre
        "attack": p["w_attack"] * np.clip(s["opp_out_speed"], -0.3, 0.3),  # m/s the rival slides outward
        "upright": p["w_up"] * s["upright_self"],
        "smooth": -p["w_smooth"] * np.sum((s["action"] - s["prev_action"]) ** 2, axis=1),
    }


TS_EVAL = dict(enabled=True, seeds="200-215", every_s=90.0, nice=10, select="best", final=True)
