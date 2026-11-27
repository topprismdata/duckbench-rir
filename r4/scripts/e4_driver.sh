#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home"
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
run() {
  local a=$1 s=$2 label="e4_a$1_s$2" cfg
  case $a in
    0) cfg=configs/sprint/baseline.py ;;
    *) cfg=configs/e4/a$a.py ;;
  esac
  if [ "$a" = "4" ]; then
    DUCK_SCRATCH=1 DUCK_DUMP_INIT_AFTER="$R/r4/artifacts/a4_init_s$s.pt" \
      uv run python train.py --config "$cfg" --label "$label" --minutes 5 --seed "$s" --model deepseek-v4.1-flash > "$R/r4/artifacts/train_$label.out" 2>&1
  else
    uv run python train.py --config "$cfg" --label "$label" --minutes 5 --seed "$s" --model deepseek-v4.1-flash > "$R/r4/artifacts/train_$label.out" 2>&1
  fi
  echo "[e4] a$a s$s ended: $(python3 -c "import json;print(json.load(open('home/runs/$label/status.json'))['state'])" 2>/dev/null || echo unknown) $(date +%H:%M:%S)"
}
run_order() { local s=$1; shift; for a in "$@"; do run "$a" "$s"; done; }
run_order 107 0 1 2 3 4
run_order 108 2 3 4 0 1
run_order 109 4 3 1 2 0
echo "[e4] ALLDONE $(date +%H:%M:%S)"
