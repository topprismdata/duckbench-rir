#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home"
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
mkdir -p "$R/r1/artifacts"
duckbench simulate sprint submissions/e1_s1/warm_start.json --seeds 8-39 --json > "$R/r1/artifacts/e1_warm_start.json"
for s in 1 2 3 4 5; do
  if [ -f "submissions/e1_s$s/submission.json" ]; then
    duckbench simulate sprint "submissions/e1_s$s/submission.json" --seeds 8-39 --json > "$R/r1/artifacts/e1_ts_s$s.json"
    echo "[eval] seed $s done"
  else
    echo "[eval] seed $s MISSING submission"
  fi
done
echo "[eval] ALLDONE"