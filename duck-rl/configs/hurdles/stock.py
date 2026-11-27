"""Hurdles (`hurdles`), from the stock walk: the stock walk (alpha_walking) on the walk node `run` of programs/hurdles.json
(walk straight down the lane at vx 0.19, head_pitch 0.6; a fall -> stock get_up -> run); walk-stand host.
Untrained on check seeds 8-39: 29.65 s, 1/32 finished (best half of 32; a miss = 30 s).
The reward pays progress down the lane while upright and reaching the finish line early; a fall ends the training
episode.
The terms (per second): progress 6 x upright^2 x clip(vx, -0.3, 0.6); lag 0.4 x exp(-(max(0, x_ref - x) / 0.15)^2) with
x_ref = min(0.19 t, 2.3 m); lateral and heading 0.1 each, paid while keeping up; a fall a one-off 0.5 s of the good-step
rate G; the finish a one-off (3 + (15 - t)) s of G; smooth -0.2 x |a - a_prev|^2. Training episodes last 15 s.
Everything here is yours to change (train.py --help lists the knobs).
"""
import numpy as np

CHALLENGE = "hurdles"
PROGRAM = "programs/hurdles.json"   # `run` walks at vx 0.19, head_pitch 0.6; a fall -> stock get_up -> run
TRAINED_NODE = "run"
WARM_START = "alpha_walking"
INIT_STD = 0.1
LR = 1e-4  # the adaptive LR, WARMUP_ITERS and the critic: the kit defaults (train.py --help)
TARGETS = [5.03, 3.42]  # the board's medium and hard rows (the run card's "time to target")

ENV = dict(start="spawn", fall="tilt45", end_on_finish=True, end_s=15.0, term_at_end=False, act_rand=False)
REWARD_PARAMS = dict(progress=6.0, vx_min=-0.3, vx_max=0.6, lag=0.4, lag_sigma=0.15, v_ref=0.19, x_ref_max=2.3,
                     lateral=0.1, lat_sigma=0.15, heading=0.1, hdg_sigma=0.3, fall_s=0.5, success_b0_s=3.0,
                     success_per_s=1.0, episode_s=15.0, action_rate=0.2)


def reward(s, p):
    """Per-second rates over the state columns (duck_rl/env.py state_doc; torso_reach's: dist, finished, finish_now,
    finish_t)."""
    G = p["progress"] * p["v_ref"] + p["lag"] + p["lateral"] + p["heading"]   # the good-step rate (alive = 0)
    vx = (s["disp"][:, 0] - s["prev_disp"][:, 0]) / 0.02                       # along the lane, m/s
    x = s["disp"][:, 0]
    keep = np.exp(-(np.maximum(0.0, np.minimum(p["v_ref"] * s["t"], p["x_ref_max"]) - x) / p["lag_sigma"]) ** 2)
    hdg = (s["hdg"] + np.pi) % (2 * np.pi) - np.pi
    return {
        "progress": p["progress"] * np.clip(s["upright"], 0.0, 1.0) ** 2 * np.clip(vx, p["vx_min"], p["vx_max"]),
        "lag": p["lag"] * keep,
        "lateral": p["lateral"] * keep * np.exp(-(s["y"] / p["lat_sigma"]) ** 2),
        "heading": p["heading"] * keep * np.exp(-(hdg / p["hdg_sigma"]) ** 2),
        "fall": -(p["fall_s"] * G / 0.02) * s["fell"],
        "finish": (p["success_b0_s"] + p["success_per_s"] * np.maximum(p["episode_s"] - s["t"], 0.0)) * G / 0.02
                  * s["finish_now"],
        "smooth": -p["action_rate"] * np.sum((s["action"] - s["prev_action"]) ** 2, axis=1),
    }


TS_EVAL = dict(enabled=True, seeds="100-115", every_s=50.0, nice=10, select="best", final=True)
