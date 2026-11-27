#!/bin/bash
set -u
R=/Users/guohongbin/1-Projects/duckbench-rir
cd "$R/duck-rl" || exit 1
export DUCKBENCH_HOME="$R/.duckbench-home"
export PATH="$R/sources/dbcli/node_modules/.bin:$PATH"
# best exports
for s in 107 108 109; do
  for a in 0 1 2 3 4; do
    sub="submissions/e4_a${a}_s${s}"
    [ -f "$sub/submission.json" ] && duckbench simulate sprint "$sub/submission.json" --seeds 8-39 --json > "$R/r4/artifacts/ts_best_a${a}_s${s}.json" && echo "[eval] best a${a} s${s} ok"
  done
done
# last checkpoints
for s in 107 108 109; do
  for a in 0 1 2 3 4; do
    d="$R/r4/last_eval/a${a}_s${s}"; mkdir -p "$d"
    if [ -f "submissions/e4_a${a}_s${s}/last.pt" ]; then
      uv run python tools/to_mlp.py "submissions/e4_a${a}_s${s}/last.pt" "$d/policy.mlp.json" --branch actor > "$d/to_mlp.log" 2>&1
      cp "submissions/e4_a${a}_s${s}/program.json" "$d/program.json"
      python3 - "$a" "$s" "$d" <<'PY'
import json,sys
a,s,d=sys.argv[1:4]
sub=json.load(open(f'submissions/e4_a{a}_s{s}/submission.json'))
sub['moves'][0]['move']='policy.mlp.json'; sub['moves'][0].pop('pin',None)
sub['notes']=f'last.pt of e4_a{a}_s{s}'
json.dump(sub, open(f'{d}/submission.json','w'), indent=1)
PY
      duckbench simulate sprint "$d/submission.json" --seeds 8-39 --json > "$R/r4/artifacts/ts_last_a${a}_s${s}.json"
      echo "[eval] last a${a} s${s} ok"
    fi
  done
done
# A4 T0 (random init per seed)
for s in 107 108 109; do
  d="$R/r4/t0_a4_s$s"; mkdir -p "$d"
  uv run python tools/to_mlp.py "$R/r4/artifacts/a4_init_s$s.pt" "$d/policy.mlp.json" > "$d/to_mlp.log" 2>&1
  cp "$R/duck-rl/programs/a3.json" "$d/program.json"
  python3 - "$s" "$d" <<'PY'
import json,sys
s,d=sys.argv[1:3]
src=json.load(open('submissions/e4_a3_s%s/submission.json'%s))
src['moves'][0]['move']='policy.mlp.json'; src['moves'][0].pop('pin',None)
src['notes']=f'A4 T0 (random init) seed {s}'
json.dump(src, open(f'{d}/submission.json','w'), indent=1)
PY
  duckbench simulate sprint "$d/submission.json" --seeds 8-39 --json > "$R/r4/artifacts/ts_t0_a4_s$s.json"
  echo "[eval] t0 a4 s$s ok"
done
echo "[eval] ALLDONE"
