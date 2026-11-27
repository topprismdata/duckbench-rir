#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home"
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
for pol in c0_s301 c0_s302 c0_s303 c1_s301 c1_s302 c1_s303 c2_s301 c2_s302 c2_s303 starter; do
  if [ "$pol" = "starter" ]; then src="$R/r8/eval/starter"; else src="$R/r7/eval/${pol}_last"; fi
  for v in v1 v2; do
    d="$R/r11/eval/${pol}_${v}"; mkdir -p "$d"
    cp "$src/policy.mlp.json" "$d/policy.mlp.json"
    cp "$src/policy.mlp.bin" "$d/policy.mlp.bin" 2>/dev/null || true
    cp "$R/r7/programs/$v.json" "$d/program.json"
    python3 - "$pol" "$v" "$d" <<'PY'
import json,sys
pol,v,d=sys.argv[1:4]
json.dump({"format":"duck-strategy/1","challenge":{"id":"steps","version":4},"program":"program.json",
           "moves":[{"nodes":["run"],"move":"policy.mlp.json"}],"notes":f"fresh-40-71 {pol} {v}"},
          open(f'{d}/strategy.json','w'), indent=1)
PY
    duckbench simulate steps "$d/strategy.json" --seeds 40-71 --json > "$R/r11/artifacts/fresh_${pol}_${v}.json" 2>/dev/null
  done
  echo "[fresh] $pol done"
done
echo "[fresh] ALLDONE"
