#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home"
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
state() { python3 -c "import json;print(json.load(open('home/runs/e3_$1$2/status.json'))['state'])" 2>/dev/null || echo pending; }
runA() { uv run python train.py --config configs/sprint/baseline.py --label e3_a$1 --minutes 5 --seed $1 --model deepseek-v4.1-flash >> "$R/r3/artifacts/train_a$1.out" 2>&1; }
runB() { DUCK_SCRATCH=1 uv run python train.py --config configs/sprint/scratch.py --label e3_b$1 --minutes 5 --seed $1 --model deepseek-v4.1-flash >> "$R/r3/artifacts/train_b$1.out" 2>&1; }
for s in 103 104 105 106; do
  odd=$(( (s - 101) % 2 ))
  if [ $odd -eq 0 ]; then order="A B"; else order="B A"; fi
  for c in $order; do
    lc=$(echo $c | tr AB ab)
    if [ "$(state $lc $s)" = "done" ]; then echo "[e3] skip $c$s (done)"; continue; fi
    echo "[e3] seed $s cond $c start $(date +%H:%M:%S)"
    if [ "$c" = "A" ]; then runA $s; else runB $s; fi
    echo "[e3] seed $s cond $c ended: $(state $lc $s) $(date +%H:%M:%S)"
  done
done
echo "[e3] ALLDONE $(date +%H:%M:%S)"
