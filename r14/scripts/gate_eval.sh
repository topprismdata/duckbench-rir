#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home" DUCKBENCH_NO_UPLOAD=1
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
sub=$1; tag=$2
duckbench simulate steps "$sub/submission.json" --seeds 8-71 --json > "$R/r14/artifacts/gate_${tag}_8_71.json" 2>/dev/null
python3 - "$R/r14/artifacts/gate_${tag}_8_71.json" "$tag" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
runs=d['seeds']; t=sorted(r['result'] if r['finished'] else 30.0 for r in runs); h=len(t)//2
print(sys.argv[2], 'best-half', round(sum(t[:h])/h,3), 'mean', round(sum(t)/len(t),3), 'finished', sum(1 for x in runs if x['finished']), '/', len(runs))
PY
