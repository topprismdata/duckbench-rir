"""Hamster wheel (`wheel`): the library move `wheel-uphill-runner-opus@1` (WARM_MLP; walk-stand host) on the walk node
`run` of programs/wheel-run.json (run up the rim at vx 0.19 from the spawn; a fall -> stock get_up -> run), the program
and node the move was trained on. configs/wheel/stock.py starts from the stock walk instead.
Untrained on check seeds 8-39: 5.33 s, 32/32 finished (best half of 32; the result is the time to one full net turn of
the wheel, 15 s if never).
The reward pays the rim's forward speed, the duck's lever on the rim, the wheel's turns and finishing early; a fall
ends the training episode.
Everything here is yours to change (train.py --help lists the knobs).
"""
import numpy as np

CHALLENGE = "wheel"
PROGRAM = "programs/wheel-run.json"
TRAINED_NODE = "run"
WARM_START = "alpha_walking"  # the stock graph (the actor's shape); the weights come from WARM_MLP
WARM_MLP = "wheel-uphill-runner-opus@1"  # its card: walk node `run`, vx 0.19
INIT_STD = 0.05
LR = 1e-4
LR_FIXED = None  # the adaptive LR
WARMUP_ITERS = 30

ENV = dict(start="spawn", fall="tilt45", end_on_finish=True, act_rand=False)
R = 0.475   # rim radius, m (info.props[0].geometry.radius)
MGR = 3.43  # duck weight torque at 90 deg, N m
FRIC = 0.12  # bearing frictionloss, N m (info.props[0].params)
DAMP = 0.9  # bearing damping, N m s / rad
RACE_S = 15.0  # the Challenge's duration = the result of a seed that never finishes (D-19)
REWARD_PARAMS = dict(w_rim=20.0, w_ang=8.0, w_metric=20.0, w_finish=5.0, w_time=0.5, fall_pen=6.0, w_smooth=0.05,
                     w_y=20.0, y_free=0.05)


def reward(s, p):
    """Per-second rates over the state columns (duck_rl/env.py state_doc; the wheel's: prop_*/wheel_*, rim_speed,
    rim_angle_deg; wheel_turns': turns_live, finished, finish_now, finish_t)."""
    rim = np.clip(s["rim_speed"], -0.2, 1.5)                         # forward rim speed, m/s
    th = np.radians(s["rim_angle_deg"])
    v_eq = np.clip(R * (MGR * np.sin(th) - FRIC) / DAMP, -0.2, 1.5)  # the rim speed the duck's lever arm would hold
    return {
        "rim": p["w_rim"] * rim,
        "angle": p["w_ang"] * v_eq,
        "metric": p["w_metric"] * s["dmetric"] / 0.02,                # sums to w_metric x the turns reached
        "finish": p["w_finish"] * s["finish_now"] / 0.02,             # one-off at the full turn
        "time": -p["w_time"] * (1.0 - s["finished"]),                 # every second before it costs
        "fall": -(p["fall_pen"] + p["w_time"] * np.maximum(0.0, RACE_S - s["t"])) * s["fell"] / 0.02,
        "smooth": -p["w_smooth"] * np.sum((s["action"] - s["prev_action"]) ** 2, axis=1),
        "lateral": -p["w_y"] * np.maximum(0.0, np.abs(s["y"]) - p["y_free"]),
    }


TS_EVAL = dict(enabled=True, seeds="100-115", every_s=50.0, nice=10, select="best", final=True)
