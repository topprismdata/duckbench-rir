"""Lazy Susan (`turntable`): the library move `turntable-starter@1` (WARM_MLP; walk-stand host) on the stand node `roll`
of programs/turntable-roll.json (clever's back-pedal round the edge with the stock walk, a stock stand `settle` from
8.5 s, then `roll` from 9.2 s to 11 s), the program and node the move was trained on. configs/turntable/stock.py starts from the stock roulade
instead.
Untrained on check seeds 8-39: 10.35 rpm (best half of 32).
Training episodes start where the program enters `roll` (a handoff bank: BANK records it into home/banks/ on the first
run) and end at the bell (10 s).
The reward pays the Challenge's score (the disc's rpm at the bell), the disc's speed from 9.8 s while the duck is on
it, and charges touching the floor before the bell.
Everything here is yours to change (train.py --help lists the knobs).
"""
import numpy as np

CHALLENGE = "turntable"
PROGRAM = "programs/turntable-roll.json"  # clever's back-pedal, settle at 8.5 s, the `roll` stand node at 9.2 s
TRAINED_NODE = "roll"
WARM_START = "roulade"  # the actor's shape; the weights come from WARM_MLP
WARM_MLP = "turntable-starter@1"  # its card: stand node `roll` from 9.2 s, after the stock back-pedal and settle
INIT_STD = 0.1
LR = 1e-4  # the adaptive LR, WARMUP_ITERS and the critic: the kit defaults (train.py --help)
TARGETS = [9.65, 11.75]  # the board's medium and hard rows (the run card's "time to target")

ENV = dict(
    start="handoff", bank="banks/turntable-settle9.2-roll.json", handoff_share=1.0,
    end_s=10.0, term_at_end=True,  # the bell: nothing after it counts
    fall="none",                   # a roll tilts the duck on purpose; only touching the floor ends an episode
    end_on_out=True,               # touching the floor before the bell ends it (score 0)
    disc_scale=None, act_rand=False,
)
BANK = dict(program=PROGRAM, nodes=TRAINED_NODE, seeds="1000-1127")  # records ENV's bank if it is missing (train.py)
REWARD_PARAMS = dict(k_score=1.0, k_live=3.0, t_live=8.0, out_penalty=3.0)


def reward(s, p):
    """Per-second reward rates {term: array[N]} over the state columns (duck_rl/env.py state_doc; the disc's in
    duck_rl/challenges/turntable.py). The trainer multiplies by the 0.02-s tick; a one-off event pays value / 0.02 on its
    tick."""
    bell = p["k_score"] * s["score"] * s["bell"] / 0.02          # the Challenge score, paid once on the bell row
    live = p["k_live"] * s["live"] * (s["t"] >= p["t_live"]) * (1.0 - s["lost"])  # shaping: disc speed near the bell
    out = -p["out_penalty"] / 0.02 * s["ended"]                  # `ended` = the out-before-the-bell termination tick
    return dict(bell=bell, live=live, out=out)


# In-loop TS eval (train.py TsEval): the newest checkpoint scored in the TS sim, in the background, while PPO trains;
# the submission is the best by the Challenge's rank key (the mean rpm, then the worst seed). Any key may be left out.
TS_EVAL = dict(enabled=True, seeds="100-115", every_s=25.0, nice=10, select="best", final=True)
