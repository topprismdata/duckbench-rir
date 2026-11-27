#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home"
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
for s in 731 732; do
  uv run python train.py --config configs/saucer/baseline.py --label r16_s$s --minutes 5 --seed $s --model deepseek-v4.1-flash > "$R/r16/artifacts/train_s$s.out" 2>&1
  echo "[saucer] seed $s done $(date +%H:%M:%S)"
done
echo "[saucer] ALLDONE"
