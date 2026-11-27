"""Saucer (`saucer`): the library move `saucer-spinner@1` (WARM_MLP; walk-stand host) on the walk node `run` of
programs/saucer-clever.json (the Saucer's clever duckscript: walk at vx 0.19, vy -0.03, wz 0.05 from the spawn; a fall ->
stock get_up -> run), the program and node the move was trained on. configs/saucer/stock.py starts from the stock walk
instead.
Untrained on check seeds 8-39: 55.59 rpm (best half of 32; the score is the top 0.5-s mean |rpm| of the disc over 0-10 s,
until the duck first touches the floor).
The reward pays the Challenge's score gained (the disc's peak rpm, weighted by the disc's mass), the disc's clockwise
spin, the duck's lever on the tilted disc and staying upright on it; it charges falls, the rim and jerky actions.
Everything here is yours to change (train.py --help lists the knobs).
"""
import numpy as np

CHALLENGE = "saucer"
PROGRAM = "programs/saucer-clever.json"
TRAINED_NODE = "run"
WARM_START = "alpha_walking"  # the stock graph (the actor's shape); the weights come from WARM_MLP
WARM_MLP = "saucer-spinner@1"  # its card: walk node `run`, vx 0.19, vy -0.03, wz 0.05
INIT_STD = 0.15
LR = 1e-4
LR_FIXED = 2e-4
WARMUP_ITERS = 30

ENV = dict(start="spawn", fall="tilt45", act_rand=False, end_on_stop=True, end_s=10.0, term_at_end=True)
TERM_AT_END = True
R_DISC = 0.7125  # the disc radius (info.props[0].geometry.radius)
REWARD_PARAMS = dict(w_peak=0.3, w_spin=0.3, mass_k=0.75, m_ref=2.5, w_drive=1.0, y_lo=-0.2, y_hi=0.5, w_alive=0.2,
                     fall_pen=1.0, w_edge=10.0, r_safe=R_DISC - 0.12, w_smooth=0.05)


def reward(s, p):
    """Per-second rates over the state columns (duck_rl/env.py state_doc; the saucer's in duck_rl/challenges/saucer.py)."""
    y = s["r"] * np.sin(s["phi"])  # the duck's lever arm on the tilted disc (m): its weight on +y turns it clockwise
    mw = (s["disc_mass"] / p["m_ref"]) ** p["mass_k"]
    on = 1.0 - s["out"]
    return {
        "peak": p["w_peak"] * mw * s["dpeak"] / 0.02,  # sums to the Challenge metric gained in the episode
        "spin": p["w_spin"] * mw * s["cw"],
        "drive": p["w_drive"] * np.clip(y, p["y_lo"], p["y_hi"]) * on,
        "alive": p["w_alive"] * s["up32"] * (1.0 - s["fell"]) * on,
        "fall": -p["fall_pen"] * s["fell"] / 0.02 * on,
        "edge": -p["w_edge"] * np.maximum(0.0, s["r"] - p["r_safe"]) * on,
        "smooth": -p["w_smooth"] * np.sum((s["action"] - s["prev_action"]) ** 2, axis=1),
    }


TS_EVAL = dict(enabled=True, seeds="100-115", every_s=50.0, nice=10, select="best", final=True)
