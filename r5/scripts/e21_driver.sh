#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home"
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
run() {
  local a=$1 s=$2 label="e21_$1_s$2"
  uv run python train.py --config "configs/e21/$a.py" --label "$label" --minutes 5 --seed "$s" --model deepseek-v4.1-flash > "$R/r5/artifacts/train_$label.out" 2>&1
  echo "[e21] $a s$s ended: $(python3 -c "import json;print(json.load(open('home/runs/$label/status.json'))['state'])" 2>/dev/null || echo unknown) $(date +%H:%M:%S)"
}
seq_for() { case $1 in 201) echo "t0 t1 t2";; 202) echo "t1 t2 t0";; 203) echo "t2 t0 t1";; esac; }
for s in 201 202 203; do for a in $(seq_for $s); do run $a $s; done; done
echo "[e21] ALLDONE $(date +%H:%M:%S)"
