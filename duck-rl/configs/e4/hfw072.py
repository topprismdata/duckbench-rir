"""Sprint (`sprint`): the library move `hurdles-fast-walker@1` (WARM_MLP; walk-stand host) on the walk node `run` of
programs/sprint-fast.json (walk straight at vx 0.19, head_pitch 0.6; a fall -> stock get_up -> run), the command the
move was trained on. configs/sprint/stock.py starts from the stock walk instead.
Untrained on check seeds 8-39: 5.68 s, 32/32 finished (best half of 32; a miss = 30 s).
The reward pays the distance gained, progress while upright, the finish line, and charges every second before the
line; a fall ends the training episode and is charged the rest of the race.
Everything here is yours to change (train.py --help lists the knobs).
"""
import numpy as np

CHALLENGE = "sprint"
PROGRAM = "programs/sprint_hp072.json"   # one walk node `run` at vx 0.19, head_pitch 0.6; get_up after a fall
TRAINED_NODE = "run"
WARM_START = "alpha_walking"            # the actor's shape; the weights come from WARM_MLP
WARM_MLP = "hurdles-fast-walker@1"      # the moves pack's fast walker (~0.75 m/s; its card: walk node, vx 0.19, head 0.6)
INIT_STD = 0.1
LR = 1e-4
LR_FIXED = 1e-4  # a fixed actor LR; WARMUP_ITERS and the critic: the kit defaults

ENV = dict(start="spawn", fall="tilt45", end_on_finish=True, act_rand=False)
RACE_S = 30.0  # the Challenge's duration = the result of a seed that never finishes (D-19)
REWARD_PARAMS = dict(w_metric=2.0, w_prog=1.0, w_finish=5.0, w_time=0.5, fall_pen=2.0, w_head=0.5, w_smooth=0.2)


def reward(s, p):
    """Per-second rates over the state columns (duck_rl/env.py state_doc; torso_reach's: dist, finished, finish_now,
    finish_t)."""
    up = s["up32"]
    fwd = (s["disp"][:, 0] - s["prev_disp"][:, 0]) / 0.02          # progress along the start heading, m/s
    return {
        "metric": p["w_metric"] * s["dmetric"] / 0.02,               # sums to w_metric x the Challenge metric reached
        "progress": p["w_prog"] * up * np.clip(fwd, -0.5, 1.0),      # dense: the metric lags 0.5 s (its window)
        "finish": p["w_finish"] * s["finish_now"] / 0.02,            # one-off at the line
        "time": -p["w_time"] * (1.0 - s["finished"]),                # every second before the line costs
        # a fall ends the training episode: charge the rest of the race (the seed's result would be 30 s at worst)
        "fall": -(p["fall_pen"] + p["w_time"] * np.maximum(0.0, RACE_S - s["t"])) * s["fell"] / 0.02,
        "heading": -p["w_head"] * (1.0 - np.cos(s["hdg"])),         # +x is the race's axis
        "smooth": -p["w_smooth"] * np.sum((s["action"] - s["prev_action"]) ** 2, axis=1),
    }


TS_EVAL = dict(enabled=True, seeds="100-115", every_s=30.0, nice=10, select="best", final=True)
