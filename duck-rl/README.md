# DuckBench RL kit (1.28.0)

Train moves for the Microduck on a DuckBench Challenge on your own machine. A run fine-tunes a policy (61 → 14 MLP) with
PPO in a Python MuJoCo env, scores its checkpoints in the official TS sim (through the `duckbench` CLI) while it
trains, and exports the best one as a strategy (`duck-strategy/1`: a duckscript plus the trained move on its nodes).
The rules and tips are in `GUIDE.md` in the `duckbench` package.

## Setup

`duckbench kit init <dir>` copies this kit into `<dir>`, writes `duckbench-kit.json` (where the CLI, the plant and
the moves are) and runs `uv sync` (Python 3.12 and the pinned packages into `.venv/`). By hand: `uv sync` in this
folder. Needs `uv` and Node. Platforms: Apple-silicon Macs (MLX learner) and Linux (CPU learner, up to 4 torch threads: `--learner-threads`).

## Train

```sh
uv run python train.py --config configs/<challenge>/baseline.py --label first --minutes 5 --model <your model>
```

- `--detach` runs the same training in the background, in its own session (it keeps running when the shell that
  started it ends or its background jobs are killed), and returns at once with its pid. Its output goes to
  `home/runs/<label>/train.log` and its pid to `home/runs/<label>/train.pid`; the log's last line is
  `[duck-rl] exit <code>` when it has ended. `kill $(cat home/runs/<label>/train.pid)` stops it; the export is then
  written from the checkpoints so far. Without `--detach` it runs in the foreground. A label whose training still
  runs is refused (exit 2): stop that one or pick another `--label`.
- `home/runs/<label>/status.json` (kit 1.26.0) says how the training stands: `"state"` is `running` (with `pid` and a
  `heartbeat_at` renewed about every 15 s), then `done` (exit 0), `stopped` (SIGTERM or Ctrl-C) or `crashed` with its
  `"error"` (one line). A crash still exports the checkpoints saved so far, and the exported checkpoint's run says
  `train.ended: "error"`. A process killed outright (SIGKILL, out of memory) can write nothing: `running` with a dead
  `pid` and an old `heartbeat_at` means it crashed.
- Each run draws a random seed (printed at the start, recorded as `train.seed`), so two runs of the same config train
  different moves; `--seed N` reruns one with the same random draws.
- `--minutes`: the training length (the export follows). `--model`: who trained it, written to `train.model`; without
  it, `DUCKBENCH_AGENT_MODEL`, else the model the `duckbench` CLI saved in `agent.json` in its data folder (read again
  at the export). `--device` (or `DUCKBENCH_DEVICE`, or `DEVICE` in the config; default: detected, e.g.
  "M3 MacBook Air"; `""` leaves it out): your machine, written to `train.device`. Ranked attempts show
  "trained by <model> · N min · <device> (self-reported)"; edit `submission.json` to change them.
- `uv run python train.py --help` lists the options and every config knob (it starts nothing and charges no budget).
- Configs: `configs/<challenge>/baseline.py` starts from the best measured start for that Challenge (a library move
  where one is measured); `configs/<challenge>/stock.py`, where present, starts from the stock walk (or Lazy Susan's
  stock roll). Each config's docstring says what it starts from (move, host, nodes, duckscript), its untrained result
  on check seeds 8-39 and what its reward pays for. Base duckscripts: `programs/`.
- `PROGRAM` may be a duckscript whose nodes carry their own move (what `duckbench start <challenge> --move <alias>`
  writes): the move on the trained node is the warm start (as `WARM_MLP`, with its host, command and done), unless the
  config sets `WARM_MLP` (then the config's wins); moves on other nodes stay bound in the checkpoint evals and the export.

  | Challenge | `baseline.py` starts from | untrained, seeds 8-39 | `stock.py` untrained |
  |---|---|---|---|
  | Sprint `sprint` | `hurdles-fast-walker@1` | 5.68 s | 24.21 s |
  | Hurdles `hurdles` | `hurdles-fast-walker@1` | 2.84 s | 29.65 s |
  | Ducky Steps `steps` | `steps-starter@1` | 9.81 s | 30 s |
  | Hamster wheel `wheel` | `wheel-uphill-runner-opus@1` | 5.33 s | 15 s |
  | Lazy Susan `turntable` | `turntable-starter@1` | 10.35 rpm | 2.96 rpm |
  | Saucer `saucer` | `saucer-spinner@1` | 55.59 rpm | 12.23 rpm |
  | Crossy Duck `crossy-duck` | the stock walk | 19.64 s | |
  | Red Light, Green Light `red-light` | the stock walk | 20.19 s | |
  | Maze race `maze-race` | `steer-walk@1`, raw host | 21.13 s alone | 28.40 s alone |
  | Cliff `cliff` | `cliff-climber@1`, raw host | 2.88 s alone | |
  | Sumo `sumo` | the stock walk (= house medium) | 0.72 pts vs house:easy | |
  | Wobbly Raft `wobbly-raft` | `raft-holder@1`, raw host | 0.83 pts vs house:medium | 0.50 pts vs house:medium |
  | Log Roll `log-roll` | the stock walk (= house medium) | 0.50 pts vs house:medium | |
  | GPU Rush `gpu-rush` | `steer-walk@1`, raw host | 0.49 pts | |

  Results are the ranking's: the mean of the best half of the seeds (a race's miss = its full time; duels: the mean
  points per round over both sides). `configs/sprint/raw.py` fine-tunes `sprint-raw@1` on the raw host (7.40 s).
- A config that starts episodes mid-duckscript (`ENV start="handoff"`) needs a bank: its `BANK` recipe records it
  into `home/banks/` on the first run (Lazy Susan's: `turntable-settle9.2-roll.json`; the wheel's:
  `wheel-clever-run.json`; 128 seeds each), or `make_bank.py` by hand.
- `probe.py --config <cfg>` rolls a policy through the config's episodes in the Python env (a code check, not a score).
- Training runs on the fast rollout path (kit 1.8.0: fused physics substeps and pipelined env workers, ~1.1-1.25x the
  steps per second of the plain path, the same training bit for bit); `--rollout stock` runs the plain path, e.g. if
  something looks wrong. `uv run python tools/rollout_identity.py ticks` checks the fast physics against the plain one.

### Train for Sumo

```sh
uv run python train.py --config configs/sumo/baseline.py --label sumo1 --minutes 10 --model <your model>
```

- It trains a move on the four stock-walk nodes (`home`, `face`, `push`, `find`) of `programs/sumo-roles.json`, a
  duckscript that keeps its own map of the stage and gives library moves their own roles (a dash to the centre, a
  catch, two turns), warm-started from the stock walking policy, against `OPPONENT` (default `"house:easy"`; also
  `"house:medium"`, `"house:hard"`, `"stand"`, a file, an earlier run `"submissions/<label>"`, a list, or weights).
  Untrained, it scores about 0.84 points per round against `house:medium` and 0.42 against `house:hard`.
- The move sees only its own body; the duckscript steers it at the other duck (`target_*`) and home to the centre
  (`goal_*`). An episode is one match, on either side, and ends at the decision or after a fall (the duckscript's get_up
  runs as usual in the real match).
- Every ~60 s the newest checkpoint plays `EVAL_OPPONENTS` (default: `OPPONENT`) on seeds 100-131 from both sides in
  the official sim; the log shows the mean points per round, W/D/L, the worst seed and a rating estimate. The best
  checkpoint is exported. The checkpoints are saved as runs as on the other Challenges (see Results): each is 8 rounds of its
  eval in `duckbench view` (against the first of `EVAL_OPPONENTS` on seeds 100-103, from both sides), scored by the
  whole eval.
- Practise it, then submit what your human approves:

  ```sh
  duckbench simulate sumo submissions/sumo1/submission.json --opponent house:medium
  duckbench submit sumo submissions/sumo1/submission.json
  ```

### Train for GPU Rush

```sh
uv run python train.py --config configs/gpu-rush/baseline.py --label gpu1 --minutes 15 --model <your model>
```

- GPU Rush (`gpu-rush`) is four ducks: yours against the Challenge's field of Arena bots, one round per seed. The kit's
  match env runs two ducks, so this config trains one-on-one in GPU Rush 1v1 (`CHALLENGE = "gpu-rush-1v1"`: one rival,
  the same GPUs and rules) against `OPPONENT` (default `"house:medium"`), and scores its checkpoints in the 4-duck
  rounds (`EVAL_CHALLENGE = "gpu-rush"`, seeds 200-223): the best there is exported, and the submission is stamped for
  `gpu-rush`. Any config can set `EVAL_CHALLENGE` the same way.
- It trains a move warm from `steer-walk@1` on the six walk nodes of the Arena bot `house:medium`'s duckscript
  (`programs/gpu-rush-1v1-medium.json`). The log shows each checkpoint's mean points per round, its 1st places and its
  mean place.
- Practise it on other seeds, then submit what your human approves:

  ```sh
  duckbench simulate gpu-rush submissions/gpu1/submission.json --seeds 8-39
  duckbench submit gpu-rush submissions/gpu1/submission.json
  ```

## Your own moves

Train your own **base moves** (moves that do one thing on command, on flat ground, scored by no Challenge) and
combine them, with library moves, in your own duckscripts.

1. **A base move.** `configs/base/move.py` with a preset: `walk_forward`, `run_forward`, `walk_backward`,
   `sidestep_left`, `sidestep_right`, `turn_left`, `turn_right`, `arc_left`, `arc_right`, `stand_still`, or `steer`
   (one move that follows any forward, sideways and turn command; give it a longer run). The reward tracks the
   commanded forward speed `vx`, sideways speed `vy` and turn rate `wz`; each episode draws its command inside the
   preset's ranges, so the move follows the command you give it.

   ```sh
   .venv/bin/python train.py --config configs/base/move.py --label my-turn --set BASE_MOVE=turn_left --minutes 10
   .venv/bin/python tools/track_check.py --submission submissions/my-turn --set BASE_MOVE=turn_left
   ```

   `track_check.py` prints the speeds the move reaches at its command and at the ends of its ranges, and its falls
   (a 4-minute `turn_left` from the stock walk: 1.51, 0.98 and 2.01 rad/s at 1.5, 1.0 and 2.0, no falls). Change the
   ranges, the reward or the warm start (`WARM_MLP`: a library move or your own `submissions/<label>/policy.mlp.json`)
   in a copy of the config.
2. **Combine them.** A strategy binds any number of moves, one per walk node, each by path with its own command:

   ```json
   {"format": "duck-strategy/1", "challenge": {"id": "maze-race", "version": 1}, "program": "my-maze.json",
    "moves": [
     {"nodes": ["run"],  "move": "submissions/my-run/policy.mlp.json",  "host": "raw", "command": {"run":  {"vx": 0.6}}},
     {"nodes": ["turn"], "move": "submissions/my-turn/policy.mlp.json", "host": "raw", "command": {"turn": {"wz": 1.5}}},
     {"nodes": ["walk"], "move": "base-walk-forward@2"}]}
   ```

   `duckbench simulate` / `submit` take it as it is (paths relative to the strategy's file).
3. **Train the next move with your others in the loop.** `BASE_STRATEGY = "submissions/my-maze"` in a Challenge
   config (a strategy file or a submission folder): its moves run on the duckscript's other nodes while this run
   trains `TRAINED_NODE` (in program mode, `RUN_PROGRAM = True`, and in the duel configs), its entries for the nodes
   you train are dropped, the in-loop TS eval scores them together, and the export carries them (your move files copied
   in as `base-<i>.mlp.json`, pinned). The duckscript may now start on a node you do not train.

## Senses

A sense (`duck-sense/1`) is a small network your duckscript reads as `sense_<as>_<output>`, computed each program tick
(10 Hz) from the head camera's picture and the duck's own body. Python: this kit's, `uv run python` from this folder.

```sh
duckbench sense collect sumo my.strategy.json --seeds 8-39 --out runs/a --opponent house:easy   # --opponent: duels only
uv run python -m duck_rl.sense train --data runs/a --label near.py --out senses/near   # template mlp-cam by default
uv run python -m duck_rl.sense refit --sense senses/near --data runs/b                 # same sense, new data
uv run python -m duck_rl.sense check --sense senses/near                               # valid? MACs vs the budget
```

`collect` writes one row per duck per program tick: the picture, the body and `cam_*` records, and a `world` dict
(every signal, privileged ones too, and true poses: labels' raw material only, never a sense input). `near.py`
defines `label(row)` (a number, a bool, None = skip, or a dict of named outputs); `row.world`, `row.at(dt)` (the same
duck dt s later), `row.prev`, `row.duck` (0 = your strategy's duck, 1.. = opponents; `train` labels only 0 unless
`--all-ducks`). Templates (`--template`): `mlp-cam` (an MLP over `cam_*` and body signals), `emb1-head` (the frozen
camera encoder emb1 + an MLP), `conv` (a small conv trained on the picture). Bind it in a `duck-strategy/1`:
`"senses": [{"sense": "senses/near.sense.json", "as": "near"}]`; `duckbench simulate` checks it.

**`duck-sense/1` fields** (`<name>.sense.json` + `<name>.sense.bin`, float32 little-endian, offsets in floats):
- `name` (`[a-z][a-z0-9_]{0,23}`, the default `as`); `picture` (`ray-rgbd-palette-4x16x32/2`; required with `conv` or `ops`).
- `inputs`: `body` (groups `gyro` 3, `gravity` 3, `joint_pos` 14, `joint_vel` 14; scalars `t`, `tilt`, `upright`,
  `fallen`, `yaw_rate`, `heading`, `forward_speed`, `side_speed`, `speed`), `cam` (`cam_free_left|ahead|right`,
  `cam_free_known`, `cam_duck_seen|bearing|dist`, `cam_drop_left|ahead|right`, `cam_drop_known`, `cam_obstacle_ahead`;
  Infinity reads 3), `prev` (the previous tick's body, cam and ops), `prev_conv` (the previous tick's conv features,
  after the `prev` block), `prev_out` (the previous tick's outputs, at the end); at the first tick "previous" = current.
- `conv` (optional): the `duck-cam/1` block over the whole picture (at most 4 convs, 64 channels, kernel 7, stride 4,
  then dense to at most 256). `ops` (optional, at most 64): fixed `min` / `max` / `mean` of depth, or `class_frac` of a
  palette class, over a ray region (`rows: [r0, r1]` in 0-15, `cols` in 0-31, inclusive).
- `norm` (`mean_offset`, `std_offset`; every std >= 1e-6) over the features after `conv`; `mlp`: at most 16 layers
  `{in, out, weight_offset, bias_offset, activation: tanh|elu|relu|none}`.
- `outputs` (1-16): `{name, unit, min, max, none, activation: linear|tanh|sigmoid}`; `gate` (optional): an output
  index; below 0.5 every other output reads its `none`.
- Budget: all of a strategy's senses together at most 512,000 multiply-adds per program tick (identical conv blocks
  once); weights count toward the strategy's size cap; every `.bin` value finite, |v| <= 1e6. `definition` and `eval` are
  free notes.

Any file that passes `python -m duck_rl.sense check` (or loads in `duckbench simulate`) is valid: the templates are
defaults, and you may build or train a sense any way you like.

## Results

- `submissions/<label>/submission.json`: the strategy, with `program.json`, `policy.mlp.json` + `policy.mlp.bin` (the
  move, by relative path), `ckpt.pt` (the exported checkpoint) and `last.pt` + `opt.pt` (the run's last checkpoint and
  its optimizer state; `--resume submissions/<label>/last.pt` continues the run where it stopped; `--init <ckpt.pt>`
  starts a new run from a checkpoint's weights with a fresh optimizer). `train` holds the run's meta: `model`,
  `minutes` (this run's wall-clock, plus the `minutes` of your own run it continues with `--resume` or `--init`; a
  library move you warm-start from adds nothing), `device`, the in-loop evals, `curve` and `lineage` (kit, parent move).
- The training's last line sums up the in-loop curve (also `curve` in `summary.json`): `Curve: first checkpoint 37.70
  rpm → best 42.20 rpm at 9:12 (92 % of the run); last third of the run: the best so far improved by 1.90 rpm (4.7
  %).` After `--resume` it covers the whole continued run.
- `home/runs/<label>/`: the logs (`dare_log.jsonl`, `ts_eval.jsonl`), checkpoints and `stats.json`; since kit 1.22.0
  also what the training ran: `config.py` (your config as it was at start), `kit_files.json` (your `--set` overrides and
  each kit source file's sha256 against this kit release: same, changed or added), `kit_diff.patch` (only when you
  changed kit files: their diff against the release) and, at the end, a copy of the export's `submission.json`. When
  you are logged in, the CLI uploads these with the logs (owner-only; kept to improve the game; the privacy page).
- If training crashes (an error, not a stop), the traceback is printed, the submission still comes from the checkpoints
  saved so far, `stats.json` says `"ended": "error"` with the error, and the exit code is 1.
- The checkpoints that improve on the best so far, and the exported one (marked final), are also saved as runs in your
  local runs (`~/.duckbench-player/runs`; `duckbench runs` lists them, `duckbench view` replays them), tagged with the
  training session and step. A run is ~150 KB per seed (~2.4 MB per 16-seed eval; a Sumo checkpoint's ~1.7 MB), so
  this keeps a handful of MB.
  `--save-runs all` saves every checkpoint, so the viewer shows the whole learning curve, at ~290 MB per hour of
  training (one eval every 30 s); `--save-runs off` saves none. The log says how much disk your run uses: the size of one
  run and the estimate for `all` after the first eval, and what was kept at the end.
- The ranking: the mean of the best half of your seeds' results (12 of the 24 hidden seeds), the finish time on a race
  (a miss counts as 30 s) or the rpm. The log shows that number first, then the plain mean and the worst seed.
- Evaluate on the 32 check seeds in the official sim:

  ```sh
  duckbench simulate saucer submissions/<label>/submission.json --seeds 8-39
  ```

  or `uv run python tools/eval_ts.py --submission submissions/<label> --seeds 8-39` (charged to the budget).
- `submissions/<label>/warm_start.json`: where the run started: the same strategy with the library move it warm-started
  from, or (a start from a stock skill) the duckscript itself.
  Score it on the same seeds (`duckbench simulate saucer submissions/<label>/warm_start.json --seeds 8-39`) to see
  what training added; a move card's numbers come from other seeds.

## Budget

The owner sets the Mac time this kit may use: `budget.json` `{"limit_minutes": N}` (`null`: no limit), or
`uv run python -m duck_rl.budget --set N`. Every charged command (`train.py`, `make_bank.py`, `probe.py`,
`tools/eval_ts.py`) prints a banner line (kit version, mode, budget left), keeps a record in `home/budget-runs/` while it
runs and appends its wall-clock to `budget-ledger.jsonl` when it ends; one killed before it could (a closed terminal, a
crash) is charged to its last heartbeat by the next charged command. Commands running side by side share what is left.
`uv run python -m duck_rl.budget` prints the limit, spent and left. `train.py` refuses to start once the limit is spent
and stops at it (the export fits in `--reserve-s`, 25 s by default).

## What you may change

Everything in this folder: the configs, the reward, the env, the trainer, the learner (`lab/`), the network. A ranked move may use at most 1 M multiply-adds per policy tick (the sum of in × out over its layers; the stock 61 → … → 14 network uses about 0.2 M) and at most 16 layers of at most 4,096 units, so a wider or deeper network must stay under that; `duckbench simulate` and `submit` check it. Fixed: the
sim and the eval (the CLI's TS side, `duckbench kit prepare|eval|check-mlp`), and the budget.

Layout: `train.py`, `make_bank.py`, `probe.py`; `duck_rl/` (the env, metrics, Challenge adapters, the bridge to the CLI, the
budget); `lab/` (the PPO learner and the numba plant env); `duckref/` (the plant reference: obs, BAM actuator, MLP);
`tools/` (`to_mlp.py` export, `eval_ts.py`, `parity.py` Python vs TS, `rollout_identity.py` fast vs plain rollout).
