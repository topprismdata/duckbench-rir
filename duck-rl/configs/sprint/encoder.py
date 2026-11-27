"""Sprint with the frozen camera encoder (kit 1.27.0, ENCODER): configs/sprint/baseline.py's duckscript, warm start and
reward, the policy reading the 61 stock inputs plus the kit encoder emb1's 64 numbers of the duck's camera picture
(a class-free embedding of the palette picture; LB-334). The encoder never trains; the export is a site camera move (README 1.27.0).
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("sprint_baseline", Path(__file__).with_name("baseline.py"))
_b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b)

CHALLENGE, PROGRAM, TRAINED_NODE, WARM_START, WARM_MLP = _b.CHALLENGE, _b.PROGRAM, _b.TRAINED_NODE, _b.WARM_START, _b.WARM_MLP
INIT_STD, LR, LR_FIXED = _b.INIT_STD, _b.LR, _b.LR_FIXED
ENV, RACE_S, REWARD_PARAMS, TS_EVAL = dict(_b.ENV), _b.RACE_S, dict(_b.REWARD_PARAMS), dict(_b.TS_EVAL)
reward = _b.reward
ENCODER = "emb1"
