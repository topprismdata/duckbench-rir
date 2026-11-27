#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home"
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
uv run python train.py --config configs/wheel/baseline.py --label r17_w20 --minutes 20 --seed 751 --init home/runs/r17_w741/ckpt.pt --model deepseek-v4.1-flash > "$R/r17/artifacts/train_w20.out" 2>&1
echo "[cont] wheel done $(date +%H:%M:%S)"
uv run python train.py --config configs/saucer/baseline.py --label r16_s20 --minutes 20 --seed 752 --init home/runs/r16_s731/ckpt.pt --model deepseek-v4.1-flash > "$R/r16/artifacts/train_s20.out" 2>&1
echo "[cont] saucer done $(date +%H:%M:%S)"
echo "[cont] ALLDONE"
