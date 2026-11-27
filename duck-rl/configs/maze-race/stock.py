"""Maze race (`maze-race`, a race duel), from the stock walk: the stock walk (alpha_walking) on the walk node `go` of
programs/maze-race-medium.json (the Arena bot house medium's duckscript: speed and turn fed every program tick from
path(0, 0), the shortest route round the walls, slower in sharp turns; a fall -> stock get_up -> BACK); walk-stand host.
Untrained on check seeds 8-39, alone (`--opponent stand`): 28.40 s to the middle (best half of 64 rounds).
Training races OPPONENT from either corner. The reward pays the outcome (+10 a win, -10 a loss or leaving `go` after a
fall), closing on the finish, being ahead, being upright, smooth actions.
Everything here is yours to change (train.py --help lists the knobs).
"""
import numpy as np

CHALLENGE = "maze-race"
PROGRAM = "programs/maze-race-medium.json"
TRAINED_NODE = "go"
WARM_START = "alpha_walking"
INIT_STD = 0.1
LR = 1e-4
LR_FIXED = 5e-5
WARMUP_ITERS = 10
GAMMA = 0.995
STAGGER_START = True  # a race is ~1500 ticks: spread the envs over it

OPPONENT = "house:medium"
EVAL_OPPONENTS = None
UNBOUND = "end"

REWARD_PARAMS = dict(w_win=10.0, w_loss=10.0, w_draw=0.0, w_unbound=10.0, w_close=2.0, w_lead=0.2, w_up=0.1,
                     w_smooth=0.05)


def reward(s, p):
    """Per-second rates over the state columns (duck_rl/env.py state_doc; the race's in duck_rl/challenges/race.py)."""
    return {
        "win": p["w_win"] * s["won"] / 0.02,
        "loss": -p["w_loss"] * s["lost"] / 0.02,
        "draw": p["w_draw"] * s["draw"] / 0.02,
        "unbound": -p["w_unbound"] * s["unbound"] / 0.02,
        "close": p["w_close"] * np.clip(s["fin_rate_self"], -0.5, 0.5),   # m/s toward the finish
        "lead": p["w_lead"] * np.clip(s["lead"], -1.0, 1.0),
        "upright": p["w_up"] * s["upright_self"],
        "smooth": -p["w_smooth"] * np.sum((s["action"] - s["prev_action"]) ** 2, axis=1),
    }


TS_EVAL = dict(enabled=True, seeds="100-115", every_s=60.0, nice=10, select="best", final=True)
