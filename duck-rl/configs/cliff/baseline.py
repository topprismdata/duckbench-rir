"""Cliff (`cliff`, a race duel): the library move `cliff-climber@1` (WARM_MLP; HOST "raw", vx 0.3 from RAW_COMMAND) on the
walk node `climb` of programs/cliff-climber.json, whose turn wz is fed every program tick from path(0, 0), the route to
the summit; stuck against a riser for 1.5 s -> `back` off 1.2 s; a fall -> stock get_up -> BACK.
Untrained on check seeds 8-39: alone (`--opponent stand`) 2.88 s to the summit (best half of 64 rounds); against
OPPONENT house:medium 1.00 points per round (64-0).
Training races OPPONENT from either end. The reward pays the outcome (+10 a win, -10 a loss), height gained while
upright, closing on the summit, being upright, smooth actions, and charges leaving `climb` (3).
Everything here is yours to change (train.py --help lists the knobs).
"""
import numpy as np

CHALLENGE = "cliff"
PROGRAM = "programs/cliff-climber.json"
TRAINED_NODE = "climb"
WARM_START = "alpha_walking"       # the actor's shape; the weights come from WARM_MLP
WARM_MLP = "cliff-climber@1"       # the moves pack's climber (its card's knobs: vx, wz)
HOST = "raw"
RAW_COMMAND = {"climb": {"vx": 0.3}}  # wz comes from the duckscript's feed every program tick
INIT_STD = 0.1
LR = 1e-4
LR_FIXED = 5e-5
WARMUP_ITERS = 10
GAMMA = 0.995
STAGGER_START = True

OPPONENT = "house:medium"
EVAL_OPPONENTS = None
UNBOUND = "end"

REWARD_PARAMS = dict(w_win=10.0, w_loss=10.0, w_draw=0.0, w_unbound=3.0, w_climb=20.0, w_close=1.0, w_up=0.1,
                     w_smooth=0.05)


def reward(s, p):
    """Per-second rates over the state columns (duck_rl/env.py state_doc; the race's in duck_rl/challenges/race.py)."""
    return {
        "win": p["w_win"] * s["won"] / 0.02,
        "loss": -p["w_loss"] * s["lost"] / 0.02,
        "draw": p["w_draw"] * s["draw"] / 0.02,
        "unbound": -p["w_unbound"] * s["unbound"] / 0.02,
        "climb": p["w_climb"] * np.clip(s["climb_rate_self"], -0.2, 0.2) * s["upright_self"],   # m/s up while upright
        "close": p["w_close"] * np.clip(s["fin_rate_self"], -0.5, 0.5),
        "upright": p["w_up"] * s["upright_self"],
        "smooth": -p["w_smooth"] * np.sum((s["action"] - s["prev_action"]) ** 2, axis=1),
    }


TS_EVAL = dict(enabled=True, seeds="100-115", every_s=60.0, nice=10, select="best", final=True)
