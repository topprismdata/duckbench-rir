"""Red Light, Green Light (`red-light`), from the stock walk: the stock walk (alpha_walking) on the three walk nodes of
programs/red-light-clever.json (the stock clever duckscript: walk to the line at vx 0.19, head_pitch 0.3, steering at
its centre; stand still on red, `freeze`, go on green; a fall -> stock get_up); walk-stand host. The env runs the whole
duckscript with the light (RUN_PROGRAM); `freeze` and `back_up` stay stock.
Untrained on check seeds 8-39: 20.19 s, 32/32 finished (best half of 32; a miss or moving on red = 30 s).
The reward pays the distance gained, progress while upright, the finish line, and charges every second before the
line; a fall or an out on red ends the training episode and is charged the rest of the race.
Everything here is yours to change (train.py --help lists the knobs).
"""
import numpy as np

CHALLENGE = "red-light"
PROGRAM = "programs/red-light-clever.json"
TRAINED_NODE = ["go", "left", "right"]  # every walk node; the first starts
RUN_PROGRAM = True
WARM_START = "alpha_walking"
INIT_STD = 0.1
LR = 1e-4  # the adaptive LR, WARMUP_ITERS and the critic: the kit defaults (train.py --help)
TARGETS = [21.33, 20.95]  # the run card's "time to target"

ENV = dict(start="spawn", fall="tilt45", end_on_finish=True, end_on_out=True, act_rand=False)
RACE_S = 30.0  # the Challenge's duration = the result of a seed that never finishes or moves on red (D-19)
REWARD_PARAMS = dict(w_metric=2.0, w_prog=1.0, w_finish=5.0, w_time=0.5, fall_pen=2.0, out_pen=2.0, w_head=0.5,
                     w_smooth=0.2)


def reward(s, p):
    """Per-second rates over the state columns (duck_rl/env.py state_doc; max_distance's: dist, finished, finish_now,
    finish_t; the light's: red_out; program mode's: ticks)."""
    up = s["up32"]
    fwd = (s["disp"][:, 0] - s["prev_disp"][:, 0]) / 0.02          # progress toward the line (+x), m/s
    rest = np.maximum(0.0, RACE_S - s["t"])
    return {
        "metric": p["w_metric"] * s["dmetric"] / 0.02,               # sums to w_metric x the Challenge metric reached
        "progress": p["w_prog"] * up * np.clip(fwd, -0.5, 1.0),      # dense: the metric lags 0.5 s (its window)
        "finish": p["w_finish"] * s["finish_now"] / 0.02,            # one-off at the line
        "time": -p["w_time"] * (1.0 - s["finished"]) * s["ticks"],   # every second before the line costs, red waits too
        # a fall or an out ends the training episode: charge the rest of the race (the seed's result is 30 s at worst)
        "fall": -(p["fall_pen"] + p["w_time"] * rest) * s["fell"] / 0.02,
        "out": -(p["out_pen"] + p["w_time"] * rest) * s["red_out"] * s["ended"] * (1.0 - s["fell"]) / 0.02,
        "heading": -p["w_head"] * (1.0 - np.cos(s["hdg"])),         # +x is straight at the line
        "smooth": -p["w_smooth"] * np.sum((s["action"] - s["prev_action"]) ** 2, axis=1),
    }


TS_EVAL = dict(enabled=True, seeds="100-115", every_s=60.0, nice=10, select="best", final=True)
