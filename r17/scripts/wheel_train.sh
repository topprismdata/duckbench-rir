#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home"
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
for s in 741 742; do
  uv run python train.py --config configs/wheel/baseline.py --label r17_w$s --minutes 5 --seed $s --model deepseek-v4.1-flash > "$R/r17/artifacts/train_w$s.out" 2>&1
  echo "[wheel] seed $s done $(date +%H:%M:%S)"
done
echo "[wheel] ALLDONE"
