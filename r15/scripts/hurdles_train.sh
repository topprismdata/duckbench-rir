#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home"
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
for s in 721 722; do
  uv run python train.py --config configs/hurdles/baseline.py --label r15_h$s --minutes 5 --seed $s --model deepseek-v4.1-flash > "$R/r15/artifacts/train_h$s.out" 2>&1
  echo "[h] seed $s done $(date +%H:%M:%S)"
done
echo "[h] ALLDONE"
