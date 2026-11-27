"""Train your own base move (kit 1.17.0): a move that does one thing on command, on flat ground, with no Challenge's
score in its reward, so you can reuse it on any Challenge and in any duckscript (bind it to a raw walk node with the
command you want). The same recipe as the house base moves (experiment 2026-10-21-base-moves): the reward
tracks the commanded forward speed vx (m/s), sideways speed vy (m/s, left positive) and turn rate wz (rad/s, left
positive), measured on the trunk in its own frame, while staying upright and smooth.

Pick the move with BASE_MOVE (a preset below), e.g.

    .venv/bin/python train.py --config configs/base/move.py --label my-turn-left --set BASE_MOVE=turn_left --minutes 20

Each preset is command-conditioned: every training episode draws its command inside the preset's ranges, so the
exported move follows the command you give its node (inside those ranges; outside them it is untested). "steer" is
the general one: one move that walks forward, back, sideways and turns, all on command (it needs a longer run).

What you get: submissions/<label>/ with policy.mlp.json (the move) and submission.json (the move on this file's
one-node duckscript). Use it in your own duckscript: a walk node with "host": "raw" and its command, e.g.

    {"nodes": ["turn"], "move": "../my-turn-left/policy.mlp.json", "host": "raw",
     "command": {"turn": {"vx": 0, "vy": 0, "wz": 1.5}}}

and combine several of your moves in one strategy (one move per node; any number of moves). To train the next move with
your earlier ones running on the duckscript's other nodes, set BASE_STRATEGY in that config (README "Your own moves").

How good it is: `tools/track_check.py --submission submissions/<label> --set BASE_MOVE=<the same>` runs it at its
command and the ends of its ranges and prints the speeds it reaches and its falls. The in-loop TS eval is off here (the
Sprint's finish time does not measure a turn), so the export is the run's last weights. Watch it:
`duckbench simulate sprint submissions/<label>/submission.json --seeds 0-3`.
Everything here is yours to change (rl-rules: only the sim, the eval and your time budget are fixed).
"""
import numpy as np

CHALLENGE = "sprint"            # a flat floor; the Sprint's score is not used (TS_EVAL off)
PROGRAM = "programs/base-move.json"
TRAINED_NODE = "move"
WARM_START = "alpha_walking"   # the stock walk's graph and weights; set WARM_MLP to start from a library move or your own
HOST = "raw"
INIT_STD = 0.1
LR = 1e-4
WARMUP_ITERS = 20

# name: (summary, command ranges per slot drawn each episode, the command the export's node gets, tracking widths)
PRESETS = {
    "walk_forward":   ("walk forward",          dict(vx=(0.25, 0.45), vy=(0, 0), wz=(0, 0)),       dict(vx=0.35), dict(vx=0.15, vy=0.10, wz=0.6)),
    "run_forward":    ("run forward",           dict(vx=(0.55, 0.8), vy=(0, 0), wz=(0, 0)),        dict(vx=0.7), dict(vx=0.20, vy=0.15, wz=0.8)),
    "walk_backward":  ("walk backward",         dict(vx=(-0.3, -0.15), vy=(0, 0), wz=(0, 0)),      dict(vx=-0.25), dict(vx=0.12, vy=0.10, wz=0.6)),
    "sidestep_left":  ("step sideways left",    dict(vx=(0, 0), vy=(0.12, 0.25), wz=(0, 0)),       dict(vy=0.2), dict(vx=0.10, vy=0.10, wz=0.6)),
    "sidestep_right": ("step sideways right",   dict(vx=(0, 0), vy=(-0.25, -0.12), wz=(0, 0)),     dict(vy=-0.2), dict(vx=0.10, vy=0.10, wz=0.6)),
    "turn_left":      ("turn left on the spot", dict(vx=(0, 0), vy=(0, 0), wz=(1.0, 2.0)),         dict(wz=1.5), dict(vx=0.12, vy=0.12, wz=0.6)),
    "turn_right":     ("turn right on the spot", dict(vx=(0, 0), vy=(0, 0), wz=(-2.0, -1.0)),      dict(wz=-1.5), dict(vx=0.12, vy=0.12, wz=0.6)),
    "arc_left":       ("walk a left curve",     dict(vx=(0.2, 0.4), vy=(0, 0), wz=(0.5, 1.1)),     dict(vx=0.3, wz=0.8), dict(vx=0.12, vy=0.12, wz=0.4)),
    "arc_right":      ("walk a right curve",    dict(vx=(0.2, 0.4), vy=(0, 0), wz=(-1.1, -0.5)),   dict(vx=0.3, wz=-0.8), dict(vx=0.12, vy=0.12, wz=0.4)),
    "stand_still":    ("stand still",           dict(vx=(0, 0), vy=(0, 0), wz=(0, 0)),             dict(), dict(vx=0.05, vy=0.05, wz=0.3)),
    "steer":          ("walk on any command",   dict(vx=(-0.25, 0.45), vy=(-0.15, 0.15), wz=(-1.5, 1.5)), dict(vx=0.3), dict(vx=0.15, vy=0.10, wz=0.5)),
}
BASE_MOVE = "walk_forward"

ENV = dict(start="spawn", end_s=6.0, fall="tilt45", act_rand=True)
TS_EVAL = dict(enabled=False)
REWARD_PARAMS = dict(w_track=3.0, w_alive=0.5, fall_pen=10.0, w_smooth=0.2, w_tilt=0.5)


def after_set():
    """train.py calls this after --set: the preset's command and ranges (and the export's command) from BASE_MOVE."""
    global RAW_COMMAND, COMMAND_RANGES
    if BASE_MOVE not in PRESETS:
        raise SystemExit(f"BASE_MOVE {BASE_MOVE!r}: one of {', '.join(PRESETS)}")
    _, ranges, cmd, sig = PRESETS[BASE_MOVE]
    RAW_COMMAND = {"move": {"vx": 0.0, "vy": 0.0, "wz": 0.0, **cmd}}
    # slots whose range is a single value stay at RAW_COMMAND's value (no draw)
    COMMAND_RANGES = {k: tuple(map(float, r)) for k, r in ranges.items() if r[0] != r[1]}
    REWARD_PARAMS.update(sig_vx=sig["vx"], sig_vy=sig["vy"], sig_wz=sig["wz"])


after_set()  # the default BASE_MOVE; train.py runs it again after --set


def reward(s, p):
    """Per-second rates over the state columns (duck_rl/env.py state_doc): track the commanded vx, vy, wz."""
    up = s["up32"]
    cmd = s["command"]
    v, w = s["lin_vel"], s["ang_vel"]
    t_vx = np.exp(-((v[:, 0] - cmd[:, 0]) / p["sig_vx"]) ** 2)
    t_vy = np.exp(-((v[:, 1] - cmd[:, 1]) / p["sig_vy"]) ** 2)
    t_wz = np.exp(-((w[:, 2] - cmd[:, 2]) / p["sig_wz"]) ** 2)
    return {
        "track": p["w_track"] * up * t_vx * t_vy * t_wz,
        "alive": p["w_alive"] * up,
        "tilt": -p["w_tilt"] * s["tilt"] ** 2,
        "fall": -p["fall_pen"] * s["fell"] / 0.02,
        "smooth": -p["w_smooth"] * np.sum((s["action"] - s["prev_action"]) ** 2, axis=1),
    }
