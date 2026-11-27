#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home"
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
for a in c0 c1 c2 c3; do
  for s in 301 302 303; do
    for ck in best last; do
      d="$R/r7/eval/${a}_s${s}_${ck}"; mkdir -p "$d"
      if [ "$ck" = "best" ]; then
        cp "submissions/t2p_${a}_s${s}/policy.mlp.json" "$d/policy.mlp.json"
        cp "submissions/t2p_${a}_s${s}/policy.mlp.bin" "$d/policy.mlp.bin"
      else
        uv run python tools/to_mlp.py "submissions/t2p_${a}_s${s}/last.pt" "$d/policy.mlp.json" --branch actor > "$d/to_mlp.log" 2>&1
      fi
      for v in v1 v2 v3 v4 v5 v6 v7; do
        cp "$R/r7/programs/$v.json" "$d/prog_$v.json"
        python3 - "$a" "$s" "$v" "$d" <<'PY'
import json,sys
a,s,v,d=sys.argv[1:5]
prog=json.load(open(f'{d}/prog_{v}.json'))
nodes=[n for n in prog['nodes'] if n!='recover']
json.dump({"format":"duck-strategy/1","challenge":{"id":"steps","version":4},"program":f"prog_{v}.json",
           "moves":[{"nodes":nodes,"move":"policy.mlp.json"}],"notes":f"T2p {a} s{s} {v}"},
          open(f'{d}/strategy_{v}.json','w'), indent=1)
PY
        duckbench simulate steps "$d/strategy_$v.json" --seeds 8-39 --json > "$R/r7/artifacts/t2p_${a}_s${s}_${ck}_${v}.json" 2>/dev/null
      done
      echo "[eval] $a s$s $ck ok"
    done
  done
done
echo "[eval] ALLDONE"
