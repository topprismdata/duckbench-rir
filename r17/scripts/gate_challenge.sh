#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home" DUCKBENCH_NO_UPLOAD=1
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
ch=$1; sub=$2; tag=$3
duckbench simulate "$ch" "$sub/submission.json" --seeds 8-71 --json > "$R/r17/artifacts/gate_${tag}.json" 2>/dev/null
python3 - "$R/r17/artifacts/gate_${tag}.json" "$tag" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
runs=d['seeds']
print(sys.argv[2], d.get('result_text'), d.get('finished'))
PY
