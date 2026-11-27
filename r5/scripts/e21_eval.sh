#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home"
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
declare -A PROG=( [fixed19]="programs/steps.json" [fixed10]="programs/steps_vx010.json" [staged]="$R/r5/programs/eval_staged.json" )
for a in t0 t1 t2; do
  for s in 201 202 203; do
    for ck in best last; do
      d="$R/r5/eval/${a}_s${s}_${ck}"; mkdir -p "$d"
      if [ "$ck" = "best" ]; then
        cp "submissions/e21_${a}_s${s}/policy.mlp.json" "$d/policy.mlp.json"
        cp "submissions/e21_${a}_s${s}/policy.mlp.bin" "$d/policy.mlp.bin"
      else
        uv run python tools/to_mlp.py "submissions/e21_${a}_s${s}/last.pt" "$d/policy.mlp.json" --branch actor > "$d/to_mlp.log" 2>&1
      fi
      for cond in fixed19 fixed10 staged; do
        cp "${PROG[$cond]}" "$d/prog_$cond.json"
        python3 - "$a" "$s" "$cond" "$d" <<'PY'
import json,sys
a,s,cond,d=sys.argv[1:5]
prog={"format":"duck-strategy/1","challenge":{"id":"steps","version":4},
      "program":f"prog_{cond}.json",
      "moves":[{"nodes":(["run","b"] if cond=="staged" else ["run"]),"move":"policy.mlp.json"}],
      "notes":f"E2.1 {a} s{s} {cond}"}
json.dump(prog, open(f'{d}/strategy_{cond}.json','w'), indent=1)
PY
        duckbench simulate steps "$d/strategy_$cond.json" --seeds 8-39 --json > "$R/r5/artifacts/e21_${a}_s${s}_${ck}_${cond}.json" 2>/dev/null
        echo "[eval] $a s$s $ck $cond ok"
      done
    done
  done
done
echo "[eval] ALLDONE"
