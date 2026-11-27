"""Players' senses (duck-sense/1, docs/senses-design.md §5a): train small networks a duckscript reads as `sense_<as>_<output>`
from `duck-sense-collect/1` data (`duckbench sense collect`). CLI: `python -m duck_rl.sense <cmd>` (spikes/duck-rl/README.md,
"Senses"):

  train  --data DIR[,DIR...] --label LABEL.py [--template mlp-cam|emb1-head|conv] --out NAME
  refit  --sense NAME --data DIR[,DIR...] [--out NAME2]
  check  --sense NAME
  synth  --out DIR [--seeds 0-7] [--ticks 150]      (a synthetic collect dir for tests)

Modules: data.py (the collect format, Row), train.py (templates, fit, export, parity), check.py (validation, MACs, id),
ref.py (the reference evaluator), synth.py (synthetic data).
"""
