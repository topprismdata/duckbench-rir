#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home"
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
state() { python3 -c "import json;print(json.load(open('home/runs/t2p_$1_s$2/status.json'))['state'])" 2>/dev/null || echo pending; }
run() {
  local a=$1 s=$2 label="t2p_${a}_s${s}" dbg=""
  case $a in c1|c2|c3) dbg="DUCK_CMD_DEBUG=1";; esac
  env $dbg uv run python train.py --config "configs/t2p/$a.py" --label "$label" --minutes 5 --seed "$s" --model deepseek-v4.1-flash > "$R/r7/artifacts/train_$label.out" 2>&1
  echo "[t2p] $a s$s ended: $(state $a $s) $(date +%H:%M:%S)"
}
seq_for() { case $1 in 301) echo "c0 c1 c2 c3";; 302) echo "c1 c3 c0 c2";; 303) echo "c2 c0 c3 c1";; esac; }
for s in 301 302 303; do for a in $(seq_for $s); do
  st=$(state $a $s)
  if [ "$st" = "done" ]; then echo "[t2p] skip $a s$s (done)"; continue; fi
  if [ "$st" = "running" ]; then
    while [ "$(state $a $s)" = "running" ]; do sleep 10; done
    echo "[t2p] $a s$s finished by earlier driver: $(state $a $s)"; continue
  fi
  echo "[t2p] $a s$s start $(date +%H:%M:%S)"
  run $a $s
done; done
echo "[t2p] ALLDONE $(date +%H:%M:%S)"
