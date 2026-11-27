#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home"
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
for s in 1 2 3 4 5; do
  echo "[driver] start seed $s $(date +%H:%M:%S)"
  uv run python train.py --config configs/sprint/baseline.py --label e1_s$s \
    --minutes 5 --seed $s --model deepseek-v4.1-flash 2>&1 | tail -3
  while :; do
    st=$(python3 -c "import json;print(json.load(open('home/runs/e1_s$s/status.json'))['state'])" 2>/dev/null || echo pending)
    [ "$st" = "running" ] || break
    sleep 10
  done
  echo "[driver] seed $s ended: $st $(date +%H:%M:%S)"
done
echo "[driver] ALLDONE $(date +%H:%M:%S)"