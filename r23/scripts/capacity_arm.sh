#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home"
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
S=913
uv run python train.py --config configs/e4/capA.py --label r23_capA --minutes 10 --seed $S --model deepseek-v4.1-flash > "$R/r23/capA.out" 2>&1
echo "[arm] A done $(date +%H:%M:%S)"
DUCK_WIDE_MLP="$R/r23/wide/base1024.mlp.json" uv run python train.py --config configs/e4/capB.py --label r23_capB --minutes 10 --seed $S --model deepseek-v4.1-flash > "$R/r23/capB.out" 2>&1
echo "[arm] B done $(date +%H:%M:%S)"
echo "[arm] ALLDONE"
