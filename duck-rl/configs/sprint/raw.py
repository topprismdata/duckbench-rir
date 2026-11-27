"""Sprint (`sprint`), raw host: the library move `sprint-raw@1` (WARM_MLP; HOST "raw": the move gets its own command
block, here vx through the card's `speed` knob) on the walk node `run` of programs/sprint.json (walk straight; a fall ->
stock get_up -> run). Training draws speed in 0.45-0.65 per episode (COMMAND_RANGES); the in-loop TS eval and the export
use RAW_COMMAND (speed 0.65).
Untrained on check seeds 8-39: 7.40 s, 32/32 finished (best half of 32; a miss = 30 s).
The reward pays progress along +x while upright, holding the commanded speed and the heading, and charges falls;
training episodes last 10 s.
Everything here is yours to change (train.py --help lists the knobs).
"""
import numpy as np

CHALLENGE = "sprint"
PROGRAM = "programs/sprint.json"
TRAINED_NODE = "run"
WARM_START = "alpha_walking"   # the stock graph (the actor's shape); the weights come from WARM_MLP
WARM_MLP = "sprint-raw@1"      # a raw card: RAW_COMMAND / COMMAND_RANGES may use its knob names (speed -> vx)
HOST = "raw"
RAW_COMMAND = {"run": {"speed": 0.65}}
COMMAND_RANGES = {"speed": (0.45, 0.65)}
INIT_STD = 0.08
LR = 1e-4
LR_FIXED = 5e-5
WARMUP_ITERS = 20

ENV = dict(start="spawn", end_s=10.0, fall="tilt45", act_rand=False)
REWARD_PARAMS = dict(w_prog=2.0, w_speed=1.0, w_head=0.5, fall_pen=10.0, w_smooth=0.2)


def reward(s, p):
    """Per-second rates over the state columns (duck_rl/env.py state_doc)."""
    up = s["up32"]
    fwd = (s["disp"][:, 0] - s["prev_disp"][:, 0]) / 0.02          # progress along the spawn heading (+x), m/s
    gap = np.maximum(s["command"][:, 0] - s["lin_vel"][:, 0], 0.0)  # one-sided: below the commanded drive only
    return {
        "progress": p["w_prog"] * up * np.clip(fwd, -0.5, 0.9),
        "speed": p["w_speed"] * up * np.exp(-(gap ** 2) / 0.25 ** 2),
        "heading": -p["w_head"] * (1.0 - np.cos(s["hdg"])),         # +x is the Challenge's axis
        "fall": -p["fall_pen"] * s["fell"] / 0.02,
        "smooth": -p["w_smooth"] * np.sum((s["action"] - s["prev_action"]) ** 2, axis=1),
    }


TS_EVAL = dict(enabled=True, seeds="100-115", every_s=30.0, nice=10, select="best", final=True)
