#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home"
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
mkdir -p "$R/r8/artifacts"
for ch in hurdles sprint; do
  for vx in 0p19 0p10; do
    for pol in c0_s301 c0_s302 c0_s303 c1_s301 c1_s302 c1_s303 c2_s301 c2_s302 c2_s303 starter; do
      if [ "$pol" = "starter" ]; then src="$R/r8/eval/starter"; else src="$R/r7/eval/${pol}_last"; fi
      d="$R/r8/eval/${pol}_${ch}_${vx}"; mkdir -p "$d"
      cp "$src/policy.mlp.json" "$d/policy.mlp.json"
      cp "$src/policy.mlp.bin" "$d/policy.mlp.bin" 2>/dev/null || true
      cp "$R/r8/programs/${ch}_${vx}.json" "$d/program.json"
      python3 - "$pol" "$ch" "$vx" "$d" <<'PY'
import json,sys
pol,ch,vx,d=sys.argv[1:5]
json.dump({"format":"duck-strategy/1","challenge":{"id":ch,"version":6 if ch=="sprint" else 4},
           "program":"program.json","moves":[{"nodes":["run"],"move":"policy.mlp.json"}],
           "notes":f"XCT {pol} {ch} vx {vx}"}, open(f'{d}/strategy.json','w'), indent=1)
PY
      duckbench simulate "$ch" "$d/strategy.json" --seeds 8-39 --json > "$R/r8/artifacts/xct_${pol}_${ch}_${vx}.json" 2>/dev/null
    done
    echo "[xct] $ch $vx done"
  done
done
echo "[xct] ALLDONE"
