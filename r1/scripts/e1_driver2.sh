#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home"
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
state() { python3 -c "import json;print(json.load(open('home/runs/e1_s$1/status.json'))['state'])" 2>/dev/null || echo pending; }
# wait for the already-running seed 1
while [ "$(state 1)" = "running" ] || [ "$(state 1)" = "pending" ]; do sleep 15; done
echo "[driver] seed 1 ended: $(state 1) $(date +%H:%M:%S)"
for s in 2 3 4 5; do
  echo "[driver] start seed $s $(date +%H:%M:%S)"
  uv run python train.py --config configs/sprint/baseline.py --label e1_s$s \
    --minutes 5 --seed $s --model deepseek-v4.1-flash > "$R/r1/artifacts/train_s$s.out" 2>&1
  echo "[driver] seed $s ended: $(state $s) $(date +%H:%M:%S)"
done
echo "[driver] ALLDONE $(date +%H:%M:%S)"
