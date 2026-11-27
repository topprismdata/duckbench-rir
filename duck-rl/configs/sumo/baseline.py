"""Sumo (`sumo`, a duel), from the stock walk: the stock walk (alpha_walking) on the four walk nodes of
programs/sumo-roles.json that the stock walk runs (home, face, push, find); walk-stand host. The duckscript keeps its own
map (it dead-reckons the centre from heading and odometry and puts the other duck on it from sight and touch) and gives
library moves their own roles on its other nodes, untrained here: base-run-forward@3 dashes to the centre (open),
base-stand-still@2 catches the duck (settle), base-turn-left@2 / base-turn-right@2 turn it to the other duck (spin_l,
spin_r); a fall -> stock get_up. programs/sumo-medium.json (the Arena bot house medium's duckscript, the start until
kit 1.29.0) is still here: PROGRAM = "programs/sumo-medium.json" with its nine walk nodes trains from it.
Untrained on check seeds 8-39 x 2 sides: against OPPONENT house:easy 0.82 points per round
(48 W 9 D 7 L of 64); against house:medium 0.84; against house:hard 0.42.
Training plays OPPONENT from either side; an episode ends at the decision or when the duckscript leaves the trained
nodes (a fall, or one of the library moves' nodes). The reward pays the outcome (+10 a win, -10 a loss), pushing the
other duck outward at contact, staying near the centre, being upright, smooth actions.
Everything here is yours to change (train.py --help lists the knobs).
"""
import numpy as np

CHALLENGE = "sumo"
PROGRAM = "programs/sumo-roles.json"
TRAINED_NODE = ["home", "face", "push", "find"]  # the nodes the stock walk runs; open, settle, spin_l, spin_r keep their library moves
WARM_START = "alpha_walking"  # the stock walking policy: untrained, the duckscript plays as written
INIT_STD = 0.1
LR = 1e-4
LR_FIXED = 5e-5  # a fixed actor LR (None: the adaptive LR)
WARMUP_ITERS = 10
GAMMA = 0.995  # ~4 s ahead at 50 Hz (the kit's default 0.99: ~2 s)
STAGGER_START = True  # spread the envs over a match's 1000 ticks

OPPONENT = "house:easy"  # or "house:medium", "stand", a file, "submissions/<label>"; a list; [(ref, weight), ...]
EVAL_OPPONENTS = None  # None: OPPONENT's; or a list, e.g. ["house:easy", "house:medium"]
UNBOUND = "skip"  # the library moves' nodes (open, settle, spin_l, spin_r) and get_up run in the world; the learner resumes after them

REWARD_PARAMS = dict(w_win=10.0, w_loss=10.0, w_draw=0.0, w_unbound=10.0, w_push=2.0, push_gap=0.25, w_centre=0.5,
                     w_up=0.1, w_smooth=0.05)


def reward(s, p):
    """Per-second rates over the state columns (duck_rl/env.py state_doc; the duel's in duck_rl/challenges/sumo.py: won,
    lost, draw, unbound, d_self, d_opp, opp_out_speed, gap, margin_self, upright_self, touching, ...)."""
    near = s["gap"] < p["push_gap"]                                   # trunks ~0.15-0.2 m apart at contact
    return {
        "win": p["w_win"] * s["won"] / 0.02,                          # one-off on the decision row
        "loss": -p["w_loss"] * s["lost"] / 0.02,
        "draw": p["w_draw"] * s["draw"] / 0.02,
        "unbound": -p["w_unbound"] * s["unbound"] / 0.02,             # a fall: the duckscript went to get_up
        "push": p["w_push"] * near * np.clip(s["opp_out_speed"], -0.5, 0.5),  # m/s the opponent moves outward
        "centre": -p["w_centre"] * s["d_self"],                       # m from the ring centre
        "upright": p["w_up"] * s["upright_self"],
        "smooth": -p["w_smooth"] * np.sum((s["action"] - s["prev_action"]) ** 2, axis=1),
    }


TS_EVAL = dict(enabled=True, seeds="100-131", every_s=60.0, nice=10, select="best", final=True)
