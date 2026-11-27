#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home"
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
mkdir -p "$R/r3/artifacts"
for s in 101 102 103 104 105 106; do
  for c in a b; do
    sub="submissions/e3_${c}${s}"
    if [ -f "$sub/submission.json" ]; then
      duckbench simulate sprint "$sub/submission.json" --seeds 8-39 --json > "$R/r3/artifacts/ts_${c}${s}.json"
      echo "[eval] ${c}${s} done"
    else
      echo "[eval] ${c}${s} MISSING"
    fi
  done
done
echo "[eval] ALLDONE"