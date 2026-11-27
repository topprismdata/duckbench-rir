#!/usr/bin/env python
"""The baseline trainer: one command = one PPO run (the move lab's lab.learn.train: rsl_rl PPO, MLX learner overlapped
with the next collect) on ChallengeEnv for any Challenge, then a submission. In an experiment, run it through the budget runner:

    budget.py run --designer <you> --label smoke --max-seconds 120 -- \
        .venv/bin/python train.py --config configs/turntable/baseline.py --label smoke

In a player's kit: `uv run python train.py --config configs/<challenge>/baseline.py --label first --minutes 5` (README.md);
add `--detach` to run it in the background in its own session (detach(): home/runs/<label>/train.log and train.pid).
`--help` starts nothing and charges no budget.

Everything here is yours to change (BRIEF.md). What it does:
  1. loads the config (a Python file; the knobs are listed in CONFIG_DOC below); compiles the duckscript with the TS
     validator for the config's Challenge (the trained node's params as the sim runs them);
  2. trains until --minutes or the budget deadline (the earlier of DUCK_RL_DEADLINE_EPOCH and the owner's limit in a
     player's kit, duck_rl/budget.py, re-read every iteration, minus a reserve for the export; the setup counts), or
     until SIGTERM (the runner's; the last periodic checkpoint is then used); checkpoints every --ckpt-every env steps;
  3. meanwhile scores the newest checkpoint in the TS sim in a niced background process (TsEval: TS_EVAL in the
     config or --ts-*; ~7 % slower training while an eval runs, ~2-3 % overall at the defaults) with the full program,
     and keeps the best by the Challenge's rank key (the server's ranking, D-19 as amended 2026-10-30, kit 1.14.0: the mean of
     the best half of the seeds' natural results, a race Challenge's finish time with a miss = 30 s or a score Challenge's
     metric, in the Challenge's direction; a tie keeps the earlier checkpoint; crowd and duel Challenges: the mean points); a checkpoint eval of a real Challenge is also saved as a run in the local run store (HOME/checkpoint-runs
     when the repo's is read-only: an experiment's pinned repo, kit 1.8.2) for the viewer
     `duckbench view` (TS_EVAL save_runs / --save-runs: "improved", the default, keeps the runs of checkpoints that
     beat the session's best so far plus the exported one; "all" every one; "off" none; README "Checkpoint runs"),
     the exported one marked train.final; the log says how much disk the runs take;
  4. writes home/runs/<label>/ (log.jsonl, dare_log.jsonl, ts_eval.jsonl, ckpt-*.pt/.onnx, stats.json) and the
     submission submissions/<label>/ (submission.json duck-strategy/1 with train.lineage, train.model (--model, else
     DUCKBENCH_AGENT_MODEL, else the CLI's agent.json), train.minutes (this command's wall-clock, plus the recorded train.minutes of the kit's
     own run that --init / --resume continues; a library warm start adds nothing), train.device (--device, else DUCKBENCH_DEVICE,
     else the config's DEVICE, else auto-detected, e.g. "M3 MacBook Air"; duck_rl/device.py) and train.seed (--seed: 1 in the monorepo,
     random per run in a player's kit), program.json, policy.mlp.json + .bin via tools/to_mlp.py, ckpt.pt = the
     checkpoint of the exported weights, last.pt + opt.pt = the run's newest checkpoint and its optimizer state (kit
     1.8.1: `--resume submissions/<label>/last.pt` continues the run where it stopped), warm_start.json = the same
     strategy with the library move WARM_MLP names (when the weights started from one); every file path relative to
     the folder), from the best-by-TS checkpoint (TS_EVAL select="best", the default) or the last weights. A spike Challenge
     (version 0, ts/spike_challenges.ts) gets the legacy duck-rl-submission/1 with the same train meta (duck-strategy/1
     wants v >= 1);
  5. prints, as its last line, the in-loop curve (kit 1.8.1; also `curve` in summary.json, stats.json and the export's
     train block): "Curve: first checkpoint <score> → best <score> at m:ss (P % of the run); last third of the run:
     the best so far improved by <change> (<%>)." in the Challenge's unit; after --resume it covers the whole continued
     run when the earlier sessions' curve is beside the checkpoint, else "(this session)". Before it (newbie test
     2026-11-02): the start -> trained line on the same seeds and opponents (N26, `start_vs_trained`: the start is scored
     once as training begins, TsEval.start_eval) and, if the machine got 3x slower than at the run's start, the
     "Mac is busy" line (N25, `busy`, duck_rl/busy.py; also printed while it trains, at most once per 5 min).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

T0 = time.time()  # the command's start: its wall-clock (the budget, train.minutes) counts the imports too
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import duck_rl  # noqa: E402  (paths, lab home)
from duck_rl import bridge, budget, busy, device, moves, provenance, status  # noqa: E402  (stdlib only)
from duck_rl import rollout  # noqa: E402  (numpy only; the fast rollout, kit 1.8.0)
from duck_rl.rank import rank_result  # noqa: E402  (the best-half mean, kit 1.14.0)

CONFIG_DOC = """config knobs (a .py file; only PROGRAM, TRAINED_NODE and reward are required):
  CHALLENGE           the Challenge id (default: the config's directory name, configs/<challenge>/x.py)
  EVAL_CHALLENGE      the Challenge the in-loop TS eval scores the checkpoints on and the export is stamped for (kit 1.11.0;
                 default: CHALLENGE): training runs on CHALLENGE, the checkpoint pick (TS_EVAL select "best") and the
                 submission's `challenge` block (id, version, world) are EVAL_CHALLENGE's, so `duckbench submit <EVAL_CHALLENGE>`
                 takes it. The duckscript must be valid there too. E.g. configs/gpu-rush: trains GPU Rush 1v1
                 (gpu-rush-1v1, against one Arena bot), scored on the 4-duck crowd Challenge gpu-rush (its field of Arena
                 bots, ts/crowd_eval.ts). EVAL_OPPONENTS only when EVAL_CHALLENGE is a duel Challenge (a crowd Challenge's field is
                 fixed)
  PROGRAM        the duckscript: a path relative to the kit (programs/<challenge>.json), or to the repo in the monorepo
                 (packages/challenges/baselines/<challenge>/clever.json). Kit 1.19.0: its walk/stand nodes may carry
                 their own move ("move", host?, command?, done?; `duckbench start <challenge> --move <alias>` writes
                 one): the trained node's move is the warm start (WARM_MLP, and HOST / RAW_COMMAND / DONE where the
                 config leaves them unset; a config WARM_MLP wins), the other nodes' moves stay bound in the TS eval
                 and the export, a raw node's feed may name the card's knobs (the CLI's loader, ts/prepare.ts --node-moves)
  TRAINED_NODE   the walk/stand node the policy runs on (a str), or a list of nodes (the first is trained from the
                 episode start; the policy is bound to all of them in the submission)
  SCHEDULE       [(t, node)]: walk -> walk switches of the same policy at fixed times (the program's `t >= T` guards);
                 those nodes are bound to the policy too
  RUN_PROGRAM    True (kit 1.9.0): the env runs the whole duckscript as the sim does (env.py program mode: its own
                 node switches, every TRAINED_NODE node driven by the policy, every other node stock inside the step);
                 a solo Challenge with a ported runtime mechanic only (duck_rl/movers.py: Crossy Duck's traffic); spawn
                 starts, no SCHEDULE, BANK, WORLDS, DONE or raw host (False: the trained node alone, as before)
  WARM_START     any stock policy of the actor's shape (alpha_walking, roulade, ...; default alpha_walking)
  WARM_MLP       a move of the actor's shape whose weights and normaliser replace WARM_START's at the start (WARM_START
                 still names the stock graph for the export): a duck-mlp/1 file (a path like PROGRAM's), a move id
                 mv_<12 hex> or an alias <slug>@<n> of the move library (`pnpm xp move list`, or the CLI's moves pack
                 in a player's kit; duck_rl/bridge.py; in an experiment the pinned repo, so the pinned library).
                 Recorded as the parent in train.lineage (README "Warm starts")
  HOST           how the trained nodes feed the move (the sim's NodePolicy host): "walk-stand" (default: the node's
                 own command block, the walk controller on a walk node, zero twist on a stand node) or "raw" (the
                 move's own 13-value block, COMMAND_SLOTS in raw policy units, verbatim: no speed map, start kick or yaw
                 loop; head and body in the obs; the node's params are ignored)
  RAW_COMMAND    raw host: {node: {slot or knob: value}} (or {node: [13 numbers]}) per trained node, the fixed
                 command; knob names come from WARM_MLP's card when it is a library ref with host.knobs; unnamed = 0
  COMMAND_RANGES raw host: {slot or knob: (lo, hi)}: command-conditioned training, those slots redrawn uniformly per
                 training episode (one draw, every trained node); the in-loop TS eval and the export use RAW_COMMAND,
                 or the range midpoints where RAW_COMMAND leaves a ranged slot unset
  DONE           the move's own done rule {after_s?, upright_s?, rolled_then_upright_s?} (either host): written into
                 the export's move entry (the node raises done, the move keeps running) and the TS eval; the env's
                 `move_done` column mirrors it, and ENV end_on_done ends the episode there (a termination)
  VISION         removed (LB-330, 2026-11-14): the ray eye is gone; a move that sees is a camera move (ENCODER)
  ENCODER        a frozen shared camera encoder's name: "emb1", the kit's encoder (64 class-free numbers; LB-334; v8.6 was
                 removed on 2026-11-16, LB-337) (kit 1.27.0; duck_rl/encoders/): the obs gains its named
                 numbers of the duck's camera picture (the site camera move's ray picture in the shared palette,
                 ray-rgbd-palette-4x16x32/2), refreshed every program tick; the encoder never trains, the policy learns its
                 MLP only. The export is the site's camera move: a 61 + camera move (the 61 inputs, then the encoder's CNN
                 as its duck-cam/1 camera block, layer 0 taking the policy's weights on the named numbers folded through
                 the encoder's head; tools/to_mlp.py). Solo Challenges, either learner (MLX too since PR #1 B5)
  INIT_STD, LR   the initial action std (0.1) and the lab's starting LR (1e-4, adaptive on a KL target)
  LR_FIXED       a fixed actor LR after the warm-up (None: adaptive, the default; the step it settles at is set by
                 DESIRED_KL); the critic's too only when CRITIC_LR is None
  WARMUP_ITERS   critic-only iterations before the actor is released, at the latest (20; WARMUP_EV may release earlier)
  learner knobs (kit 1.6.2; defaults since kit 1.8.0: an adaptive actor step at KL 0.03 + a separate critic, better
  than or equal to kit 1.7's on five Challenges; kit 1.7's: DESIRED_KL 0.01, CRITIC_LR None, CRITIC_EPOCHS None,
  VALUE_CLIP 0.2, WARMUP_EV None, VALUE_BIAS_INIT False, WARMUP_ITERS 10; README "Learner knobs"):
  DESIRED_KL     the adaptive LR's KL target per minibatch (0.03): the trust-region size per update
  DESIRED_KL_END experimental (2026-10-11): the KL target falls linearly from DESIRED_KL to this over the run's
                 --minutes (wall-clock share, steps of 0.005; dkl in dare_log.jsonl); None (the default): no schedule
  CRITIC_LR      the critic's own fixed LR after the warm-up (5e-4; None: the actor's LR)
  CRITIC_EPOCHS  critic epochs per update (10; None: the actor's 5; more = extra critic-only epochs; MLX learner only:
                 the CPU learner runs None unless the config sets it)
  VALUE_CLIP     the value-loss clip around the rollout's values (None: unclipped; CPU learner: 0.2 or None)
  ROLLOUT_STEPS  steps per env per iteration (64; the batch = envs x this)
  GAMMA, LAMBDA  the discount and the GAE lambda (0.99, 0.95)
  STD_FLOOR      the learned action std never goes below this (None)
  STAGGER_START  each env's first episode is cut at a random length, so the envs run out of phase (False)
  WARMUP_EV      release the actor once the critic's explained variance on the last batch reaches this, at the latest
                 after WARMUP_ITERS (0.7; None: always WARMUP_ITERS)
  VALUE_BIAS_INIT  a fresh critic's output bias starts at the first rollout's mean return (True)
  LR_ANNEAL      "linear": LR_FIXED (and CRITIC_LR) fall linearly to 0 at the run's end; "cos3": LR_FIXED runs three
                 equal cosine cycles down to 0.1 x LR_FIXED with warm restarts, the critic at a constant CRITIC_LR (else
                 LR_FIXED); the cycle index goes into dare_log.jsonl and ts_eval.jsonl (None; needs LR_FIXED)
  TARGETS        absolute thresholds of the Challenge's natural result for the run card (summary.json run_card), e.g.
                 [30, 40, 45] rpm; on a race, finish times (reached when the mean is at or below)
  ENV            ChallengeEnv cfg (duck_rl/env.py): start, bank, handoff_share, end_s, term_at_end, fall, down_s,
                 max_tilt_deg, end_on_out, end_on_done, out_before_s, act_rand, joint_noise, disc_/wheel_/prop_scale,
                 ..._rpm0
  BANK           the recipe that records a missing ENV bank on first use: dict(program, nodes, seeds="1000-1255")
                 (make_bank.py's arguments; stock policies; written under home/)
  END_S, TERM_AT_END, FALL, OUT_BEFORE_S   the same as ENV keys, top level (they win over ENV)
  reward(s, p)   per-second reward rates {term: array[N]} over the state columns (STATE_DOC of the env)
  REWARD_PARAMS  p
  TS_EVAL        in-loop TS eval knobs (TS_EVAL_DEFAULTS)
  DEVICE         the export's train.device when neither --device nor DUCKBENCH_DEVICE is set (default: auto-detected,
                 e.g. "M3 MacBook Air"; "" leaves it out)
  ROLLOUT        "fast" (the default: fused physics substeps + the pipelined env workers, duck_rl/rollout.py; the same
                 training bit for bit, ~1.1-1.25x the steps/s) or "stock" (the kit 1.7 path); --rollout wins over it
  a duel Challenge (Sumo, PvP; duck_rl/duel_env.py): the policy plays the learner's duck (side A) on TRAINED_NODE (walk or
  stand nodes; the duckscript's other nodes run stock) against a frozen opponent; an episode is a match from the spawn,
  the side (slot) drawn 50/50, and ends at the decision or when the learner leaves its trained nodes (UNBOUND):
  OPPONENT       who the learner plays, drawn per training episode: "house:easy" (the default), "house:medium",
                 "house:hard" (the Arena bots the Challenge ships), "stand" (a duck that stands still), or a path (like
                 PROGRAM's) to a duckscript (duck-program/1 or /2) or duck-strategy/1 file or a kit submission dir
                 (submissions/<label>: a frozen earlier policy); or a list of these (drawn uniformly) or of (ref,
                 weight) pairs (0 < weight <= 1e6)
  EVAL_OPPONENTS the in-loop TS eval's opponents (ts/eval.ts --opponent, a ref or a list of refs, each once and with
                 no weights; default: OPPONENT's); its rank key is [the mean points per round over them all, the worst
                 seed's]; TS_EVAL defaults to seeds "100-115" and every_s 60 on a duel (16 seeds x 2 sides x each
                 opponent)
  UNBOUND        when the learner's active node is not a TRAINED_NODE (e.g. recover, get_up after a fall): "end" (the
                 default: the episode ends there, a termination with the `unbound` event; the TS eval and the server
                 run those nodes as usual) or "skip" (a fall still ends it so; any other untrained node runs stock inside
                 the step until the learner's next trained tick: for TRAINED_NODE on a subset of the walk nodes)
  HOST "raw", RAW_COMMAND, COMMAND_RANGES  as for a solo Challenge (kit 1.10.0): the trained nodes bound raw (e.g. WARM_MLP
                 "steer-walk@1" with its speed / wz knobs per node), one draw of the ranged slots per training episode
                 for every trained node
  not for a duel: SCHEDULE, BANK and ENV (a match starts at the spawn and its rules end it), DONE, WORLDS, END_S /
                 TERM_AT_END / FALL / OUT_BEFORE_S, TS_EVAL variants / select_key"""
SAVE_RUNS_MODES = ("improved", "all", "off")


def build_parser() -> argparse.ArgumentParser:
    """The command line (defined before the heavy imports: `train.py --help` prints it at once and charges no budget)."""
    ap = argparse.ArgumentParser(epilog=CONFIG_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--minutes", type=float, default=1e9, help="training wall-clock cap (the budget deadline also caps)")
    ap.add_argument("--reserve-s", type=float, default=25.0, help="seconds kept before the deadline for the export")
    ap.add_argument("--workers", type=int, default=None, help="env worker processes (default: one per performance core "
                    "on Apple silicon, else 8)")
    ap.add_argument("--per-worker", type=int, default=None, help="envs per worker (default: ~128 envs in all, a batch of "
                    "~8192 steps)")
    ap.add_argument("--nice", type=int, default=10, help="lower the training's CPU priority by this much (0 = not at "
                    "all), so your apps stay responsive while it trains")
    ap.add_argument("--seed", type=int, default=None, help="the training seed, recorded in train.seed (the same seed "
                    "reruns a run with the same random draws); default 1 in the monorepo, a random one per run in a "
                    "player's kit (so players who follow the same config train different moves)")
    ap.add_argument("--init", default=None, help="start from a checkpoint .pt's weights, std and critic as a new run: a "
                    "fresh optimizer, the LR restarts and the critic warms up again (to continue a run, --resume)")
    ap.add_argument("--resume", default=None, help="continue a run from a checkpoint .pt as one run (where an earlier "
                    "run stopped: submissions/<label>/last.pt): like --init, plus the "
                    "adaptive LR, no critic warm-up when the checkpoint has a critic, and the optimizer's state (Adam "
                    "moments and steps of the actor and the critic) from its sidecar opt-<steps>.pt, which a kit >= 1.6.2 "
                    "MLX run keeps for its newest and final checkpoint only (else a fresh Adam, with a note)")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="NAME=VALUE",
                    help="override a config knob after the config file ran, e.g. --set DESIRED_KL=0.01 --set "
                    "CRITIC_LR=None (VALUE: a Python literal; repeatable); recorded in stats.json and the run card")
    ap.add_argument("--ckpt-every", type=int, default=250_000, help="checkpoint (and in-loop TS eval candidate) every N env steps")
    ap.add_argument("--learner", default=None, choices=["mlx", "cpu"],
                    help="the PPO update: mlx (Apple silicon) or cpu (rsl_rl); default mlx where it can run, else cpu")
    ap.add_argument("--learner-threads", type=int, default=None,
                    help="torch threads of the CPU learner (default: up to 4, from the cores the env workers leave free; "
                         "torch's own default takes every core, on top of the workers)")
    ap.add_argument("--no-ts-eval", action="store_true", help="no in-loop TS eval (submit the last weights)")
    ap.add_argument("--ts-seeds", default=None, help="in-loop TS eval seeds (config TS_EVAL['seeds'])")
    ap.add_argument("--ts-every-s", dest="ts_every_s", type=float, default=None, help="min seconds between TS evals")
    ap.add_argument("--ts-select", default=None, choices=["best", "last"], help="submit the best-by-TS or the last weights")
    ap.add_argument("--model", default=None, help="the model that trains this (the export's train.model, shown on ranked "
                    "attempts; default DUCKBENCH_AGENT_MODEL, else the one the CLI remembered, else none): 1-64 chars of [A-Za-z0-9 ._@/+:-]")
    ap.add_argument("--device", default=None, help="the training machine (the export's train.device, shown on ranked attempts; "
                    "default DUCKBENCH_DEVICE, else the config's DEVICE, else auto-detected, e.g. \"M3 MacBook Air\"; \"\" "
                    "leaves it out): 1-32 chars of [A-Za-z0-9 ._@/+:-]")
    ap.add_argument("--save-runs", default=None, choices=SAVE_RUNS_MODES, help="which checkpoint evals stay as runs in the "
                    "local run store, for the viewer `duckbench view` (config TS_EVAL['save_runs']): improved (the default: "
                    "the checkpoints that beat the session's best so far, plus the exported one), all (every checkpoint: "
                    "~150 KB per 30-s seed, ~2.4 MB per 16-seed eval, ~290 MB per hour at one eval every 30 s), off; the "
                    "log prints the measured size and the estimate for your run")
    ap.add_argument("--no-save-runs", dest="save_runs", action="store_const", const="off", help="= --save-runs off")
    ap.add_argument("--rollout", default=None, choices=rollout.MODES, help="the rollout path (config ROLLOUT; default "
                    "fast: fused physics substeps + pipelined env workers, the same training bit for bit; stock: the kit "
                    "1.7 path, e.g. to rule the fast path out when something looks wrong)")
    ap.add_argument("--detach", action="store_true", help="run the training in the background, detached from this "
                    "shell (its own session: it survives a harness that ends the shell or kills its background jobs): "
                    "prints the pid and returns at once; the output goes to home/runs/<label>/train.log, the pid to "
                    "train.pid there, and the log's last line is \"[duck-rl] exit <code>\" when it has ended")
    return ap


DETACHED = os.environ.pop("DUCK_RL_DETACHED", None)  # set by --detach for its child: the run dir the parent made
# kit 1.26.0: this training's status.json (duck_rl/status.py; run() makes it) and the crash run() caught, for the end
STATUS: dict = {"s": None, "error": None}


def _repoint(args: list[str], old: Path, new: Path) -> list[str]:
    """args with an --init / --resume path inside `old` (the run dir --detach moved aside) pointed into `new`."""
    out, prev = [], None
    for x in args:
        for flag in ("--init", "--resume"):
            if prev == flag or x.startswith(flag + "="):
                pre, val = (flag + "=", x[len(flag) + 1:]) if x.startswith(flag + "=") else ("", x)
                with contextlib.suppress(ValueError):
                    x = pre + str(new / Path(val).expanduser().resolve().relative_to(old))
        out.append(x)
        prev = x
    return out


def refuse_live(out: Path, label: str) -> None:
    """B1 (kit 1.26.0): a training whose run dir is `out` still runs (its train.pid or status.json pid is alive): stop
    here (exit 2) rather than move its dir aside under it."""
    pid = status.live_pid(out)
    if pid is not None:
        print(f"[duck-rl] a training with label {label!r} is still running, pid {pid}: stop it (kill {pid}) or use "
              "another --label", file=sys.stderr, flush=True)
        raise SystemExit(2)


def detach(argv: list[str]) -> int:
    """--detach: this same command started again as a child in its own session (setsid: no controlling terminal, not in
    the shell's process group or job table, so a harness that ends the shell or kills its background jobs leaves it
    running), its output in home/runs/<label>/train.log and its pid in train.pid there; returns at once. The child
    trains (and charges the budget) exactly as a foreground run; this command charges nothing."""
    a = build_parser().parse_args(argv)  # the usual checks (a missing --config or --label) before anything starts
    args = [x for x in argv if x != "--detach"]
    out = duck_rl.RUNS / a.label
    refuse_live(out, a.label)
    if out.exists():  # moved aside as a foreground run does, here: the child keeps the dir that holds its log
        old = out.resolve()
        aside = out.with_name(f"{a.label}.old-{time.strftime('%H%M%S')}")
        shutil.move(str(out), str(aside))
        args = _repoint(args, old, aside.resolve())
    out.mkdir(parents=True)
    log_path, pid_path = out / "train.log", out / "train.pid"
    with open(log_path, "ab") as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *args], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
                                 env=dict(os.environ, DUCK_RL_DETACHED=str(out), PYTHONUNBUFFERED="1"))
    pid_path.write_text(f"{child.pid}\n")
    # kit 1.26.0: status.json says "running" from now on (the child rewrites it, beats and writes the end)
    status.Status(out, a.label, pid=child.pid, started_at=status.now_iso()).start(heartbeat=False)

    def rel(f: Path) -> str:
        with contextlib.suppress(ValueError):
            return str(f.resolve().relative_to(Path.cwd().resolve()))
        return str(f)
    print(f"[duck-rl] training {a.label!r} detached: pid {child.pid}\n"
          f"  log:   {rel(log_path)} (its last line is \"[duck-rl] exit <code>\" when the training has ended)\n"
          f"  state: {rel(out / status.FILE)} (\"state\": running, then done, stopped or crashed with its \"error\"; "
          "running with a dead pid = crashed)\n"
          f"  watch: tail -n 5 {rel(log_path)}\n"
          f"  alive: kill -0 $(cat {rel(pid_path)}) && echo running\n"
          f"  stop:  kill $(cat {rel(pid_path)})   (training stops and the export is written from the checkpoints so far)", flush=True)
    return 0


if __name__ == "__main__":
    if {"-h", "--help"} & set(sys.argv[1:]):  # no heavy imports and no budget charge for the help
        build_parser().print_help()
        sys.exit(0)
    if "--detach" in sys.argv[1:] and DETACHED is None:
        sys.exit(detach(sys.argv[1:]))
    budget.begin("train", T0)  # the budget record goes up before the heavy imports (a run killed in them is charged)

import functools  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import lab.learn as L  # noqa: E402
from lab.spec import Move  # noqa: E402
from duck_rl.controller import COMMAND_SLOTS, command_vector  # noqa: E402
from duck_rl.env import ChallengePool, make_env, make_unpack  # noqa: E402
from duck_rl.duel_env import END_REASONS as DUEL_END_REASONS  # noqa: E402  (a duel Challenge's episode ends)

MODEL_RE = re.compile(r"[A-Za-z0-9 ._@/+:-]{1,64}")  # train.model as the server takes it (schema trainModelName)
STOP = {"sig": None}

# dare_log.jsonl (kit 1.6.2): the episode-end reasons counted per iteration (duck_rl/env.py ChallengeEnv.end_reason; a duel
# Challenge's: DUEL_END_REASONS, duck_rl/duel_env.py DuelEnv.end_reason, plus its mean points and per-opponent tally), and
# the log.jsonl fields copied through (timing, the update's diagnostics: lab/learn.py UPD_KEYS + the batch's stats)
END_REASONS = ("timeout", "fall", "out", "finish", "stop", "done")
DLOG_COPY = ("sps_collect", "t_collect", "t_update", "t_wait", "warmup", "surrogate", "w_max", "iw_capped",
             "ratio_clamped", "std_min", "std_max", "loadavg1", "upd_it", "upd_warmup", "upd_lr", *L.UPD_KEYS, "ev", "v_mean", "ret_mean",
             "ret_std", "adv_std_raw", "value_bias_shift", "lr_cycle", "mlx_fallback")

# the learner's config knobs and their defaults (CONFIG_DOC). Kit 1.8.0: the arm `u` of docs/research/training-speedup/
# README.md section 10 (the adaptive actor step at DESIRED_KL 0.03 + the separate critic: CRITIC_LR 5e-4, 10 epochs, no
# value clip, the EV warm-up, the bias init): better than kit 1.7's defaults on Saucer, Sprint, Hurdles and the wheel,
# equal on Lazy Susan, worse nowhere; untested on Sumo. Kit 1.7's (bit for bit the kit before 1.6.2): KIT17_LEARNER.
LEARNER_KNOBS = dict(INIT_STD=0.1, LR=1e-4, LR_FIXED=None, WARMUP_ITERS=20, DESIRED_KL=0.03, CRITIC_LR=5e-4,
                     CRITIC_EPOCHS=10, VALUE_CLIP=None, ROLLOUT_STEPS=64, GAMMA=0.99, LAMBDA=0.95, STD_FLOOR=None,
                     STAGGER_START=False, WARMUP_EV=0.7, VALUE_BIAS_INIT=True, LR_ANNEAL=None)
KIT17_LEARNER = dict(WARMUP_ITERS=10, DESIRED_KL=0.01, CRITIC_LR=None, CRITIC_EPOCHS=None, VALUE_CLIP=0.2,
                     WARMUP_EV=None, VALUE_BIAS_INIT=False)  # --set these to train as kit 1.7 did


def learner_knobs(cfg_mod, learner: str) -> dict:
    """The config's learner knobs with the defaults filled in, checked (SystemExit on a bad one)."""
    k = {n: getattr(cfg_mod, n, d) for n, d in LEARNER_KNOBS.items()}

    def bad(name: str, want: str):
        raise SystemExit(f"{name} = {k[name]!r}: {want}")

    def num(name: str, lo: float | None = None, hi: float | None = None, none: bool = False, integer: bool = False):
        v = k[name]
        if v is None and none:
            return
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (integer and not float(v).is_integer()) \
                or not math.isfinite(v) or (lo is not None and v <= lo) or (hi is not None and v > hi):
            bad(name, f"a{'n integer' if integer else ' number'}{f' > {lo}' if lo is not None else ''}"
                      f"{f' and <= {hi}' if hi is not None else ''}{' or None' if none else ''}")
        k[name] = int(v) if integer else float(v)
    num("INIT_STD", 0)
    num("LR", 0)
    num("LR_FIXED", 0, none=True)
    num("WARMUP_ITERS", -1, integer=True)
    num("DESIRED_KL", 0)
    num("CRITIC_LR", 0, none=True)
    num("CRITIC_EPOCHS", 0, none=True, integer=True)
    num("VALUE_CLIP", 0, none=True)
    num("ROLLOUT_STEPS", 7, integer=True)
    num("GAMMA", 0, 1)
    num("LAMBDA", -1e-12, 1)
    num("STD_FLOOR", 0, none=True)
    num("WARMUP_EV", None, 1, none=True)
    for name in ("STAGGER_START", "VALUE_BIAS_INIT"):
        if not isinstance(k[name], bool):
            bad(name, "True or False")
    if k["LR_ANNEAL"] not in L.LR_ANNEAL_MODES:
        bad("LR_ANNEAL", "None, 'linear' or 'cos3'")
    if k["LR_ANNEAL"] and k["LR_FIXED"] is None:
        bad("LR_ANNEAL", "needs LR_FIXED (the adaptive LR has no schedule to anneal)")
    if learner == "cpu" and k["CRITIC_EPOCHS"] is not None and not hasattr(cfg_mod, "CRITIC_EPOCHS"):
        k["CRITIC_EPOCHS"] = None  # the default's extra critic epochs are MLX-only: the CPU learner runs its 5
        print("[duck-rl] note: --learner cpu runs the critic's 5 epochs (the default CRITIC_EPOCHS 10 is MLX only)",
              flush=True)
    if learner == "cpu" and (k["CRITIC_EPOCHS"] is not None or k["VALUE_CLIP"] not in (None, 0.2)):
        raise SystemExit("CRITIC_EPOCHS and a VALUE_CLIP other than 0.2 / None need the MLX learner (--learner mlx)")
    return k


KL_STEP = 0.005  # DESIRED_KL_END: the scheduled target moves in steps of this (mlx_ppo compiles one step per target)


class KlSchedule:
    """Experimental (docs/research/training-speedup/README.md section 14, 2026-10-11): DESIRED_KL_END anneals the
    adaptive LR's KL target from DESIRED_KL to DESIRED_KL_END, linearly in the elapsed share of the run's --minutes,
    rounded to KL_STEP (each distinct target compiles its own MLX minibatch step). install() wraps lab.learn.Updater so
    every update sends the current target; advance(frac) (train.py's log_fn, once per iteration) moves it. Unset (None),
    nothing of this runs: the training is the kit's as before, bit for bit."""

    def __init__(self, start: float, end: float):
        self.start, self.end, self.now = start, end, start

    def advance(self, frac: float) -> float:
        frac = min(1.0, max(0.0, frac))
        self.now = round(round((self.start + (self.end - self.start) * frac) / KL_STEP) * KL_STEP, 6)
        self.now = max(self.now, KL_STEP)
        return self.now

    def install(self) -> None:
        sched, base = self, L.Updater

        class KlSchedUpdater(base):
            def update(self, *args, **kw):
                d = sched.now
                self.knobs["dkl"] = d  # (the MLX learner reads its own copy of the knobs; the CPU learner rsl_rl's)
                if getattr(self, "mlx", None) is not None:
                    self.mlx.knobs["dkl"] = d
                if hasattr(self.alg, "desired_kl"):
                    self.alg.desired_kl = d
                return super().update(*args, **kw)

        L.Updater = KlSchedUpdater


def kl_schedule(cfg_mod, knobs: dict) -> KlSchedule | None:
    """The config's DESIRED_KL_END (experimental; None, the default: no schedule), checked."""
    end = getattr(cfg_mod, "DESIRED_KL_END", None)
    if end is None:
        return None
    if isinstance(end, bool) or not isinstance(end, (int, float)) or not math.isfinite(end) or end <= 0:
        raise SystemExit(f"DESIRED_KL_END = {end!r}: a number > 0 or None")
    if knobs["LR_FIXED"] is not None:
        raise SystemExit("DESIRED_KL_END needs the adaptive LR (LR_FIXED None)")
    return KlSchedule(knobs["DESIRED_KL"], float(end))


def apply_overrides(cfg_mod, sets: list[str]) -> dict:
    """--set NAME=VALUE (repeatable): a config attribute set after the config file ran (VALUE a Python literal:
    0.03, None, True, 'linear', [30, 40]; anything else is taken as a string). -> {NAME: value}."""
    import ast
    out = {}
    for s in sets or []:
        name, sep, raw = s.partition("=")
        name = name.strip()
        if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise SystemExit(f"--set {s!r}: want NAME=VALUE with an upper-case config name")
        try:
            val = ast.literal_eval(raw.strip())
        except (ValueError, SyntaxError):
            val = raw.strip()
        setattr(cfg_mod, name, val)
        out[name] = val
    return out


def recipe_hash(knobs: dict, extra: dict, cfg_path: Path) -> str:
    """sha256 of the effective learner knobs + the run shape (extra) + the config file's bytes: one id per recipe."""
    import hashlib
    h = hashlib.sha256(json.dumps(dict(knobs=knobs, **extra), sort_keys=True, default=str).encode())
    h.update(cfg_path.read_bytes())
    return "sha256:" + h.hexdigest()[:16]


class StopTraining(Exception):
    pass


def _on_term(signum, _frame):
    STOP["sig"] = signum


class ChallengeVecEnv(L.PoolVecEnv):
    """PoolVecEnv + per-env episode returns."""

    CURRENT = None

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.ret = np.zeros(self.num_envs)
        self.done_returns: list[float] = []
        ChallengeVecEnv.CURRENT = self

    def step(self, actions):
        obs, rew, dones, extras = super().step(actions)
        self.ret += rew.numpy().astype(np.float64)
        d = dones.numpy().astype(bool)
        if d.any():
            self.done_returns.extend(self.ret[d].tolist())
            self.ret[d] = 0.0
        return obs, rew, dones, extras

    def reset_acc(self):
        super().reset_acc()
        if hasattr(self.pool, "reset_timing"):  # ChallengePool: the workers' busy time per iteration
            self.pool.reset_timing()

    def rollout_act(self, alg, rollout_steps: int):
        """lab/learn.py's act: the fast rollout's Pipeliner when the pool pipelines, else PPO.act (duck_rl/rollout.py)."""
        return rollout.make_act(alg, self.pool, rollout_steps)


class ChallengeMove(Move):
    def __init__(self, cfg: dict, **kw):
        super().__init__(**kw)
        self.cfg = cfg

    def env_cfg(self) -> dict:
        return dict(self.cfg)


def load_config(path: Path):
    spec = importlib.util.spec_from_file_location("duck_rl_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def config_challenge(cfg_mod, path: Path) -> str:
    # a config written before the 2026-10-22 rename names it DARE (and EVAL_DARE): read both  rename-challenge:keep
    return getattr(cfg_mod, "CHALLENGE", None) or getattr(cfg_mod, "DARE", None) or path.resolve().parent.name  # rename-challenge:keep


def config_eval_challenge(cfg_mod, challenge: str) -> str:
    """EVAL_CHALLENGE (kit 1.11.0): the Challenge the checkpoints are scored on and the export is stamped for; default CHALLENGE."""
    return getattr(cfg_mod, "EVAL_CHALLENGE", None) or getattr(cfg_mod, "EVAL_DARE", None) or challenge  # rename-challenge:keep


def resolve(p: str | Path) -> Path:
    """A config path: absolute, else relative to the kit, else to the repo (stock programs; the monorepo), else to HOME
    (banks)."""
    p = Path(p).expanduser()
    if p.is_absolute():
        return p
    for base in (HERE, duck_rl.REPO, duck_rl.HOME):
        if base is not None and (base / p).exists():
            return (base / p).resolve()
    return (HERE / p).resolve()


def trained_nodes(cfg_mod) -> list[str]:
    tn = cfg_mod.TRAINED_NODE
    nodes = [tn] if isinstance(tn, str) else list(tn)
    for _, nid in getattr(cfg_mod, "SCHEDULE", []):
        if nid not in nodes:
            nodes.append(nid)
    return nodes


def base_moves(cfg_mod, nodes: list[str]) -> list[dict]:
    """BASE_STRATEGY (kit 1.17.0): a duck-strategy/1 file or submission dir (e.g. "submissions/my-walk", an earlier run of
    yours on the same duckscript) whose moves run on the duckscript's other nodes while this run trains TRAINED_NODE, and
    go into the export with it, so your moves train and are scored together. Its entries' nodes that this run trains are
    dropped (the new move replaces them there); a move path resolves against the strategy's folder (made absolute), a
    library ref (mv_ id, alias) stays. -> duck-strategy/1 move entries {nodes, move, host?, command?, done?, pin?}."""
    ref = getattr(cfg_mod, "BASE_STRATEGY", None)
    if not ref:
        return []
    f = resolve(ref)
    if f.is_dir():
        f = f / "submission.json"
    if not f.exists():
        raise SystemExit(f"BASE_STRATEGY {ref!r}: no file {f} (want a duck-strategy/1 file or a submission dir)")
    sub = json.loads(f.read_text())
    entries = sub.get("moves")
    if sub.get("format") == "duck-rl-submission/1":  # the legacy envelope: policies [{nodes, mlp}]
        entries = [dict(nodes=e["nodes"], move=e["mlp"]) for e in sub.get("policies", [])]
    if sub.get("format") not in (moves.STRATEGY_FORMAT, "duck-rl-submission/1") or not isinstance(entries, list):
        raise SystemExit(f"BASE_STRATEGY {f}: not a duck-strategy/1 file (format {sub.get('format')!r})")
    out = []
    for e in entries:
        keep = [n for n in e.get("nodes") or [] if n not in nodes]
        if not keep:
            continue
        mv = str(e["move"])
        if not moves.is_library_ref(mv):
            path = Path(mv).expanduser()
            path = (path if path.is_absolute() else f.parent / path).resolve()
            if not path.exists():
                raise SystemExit(f"BASE_STRATEGY {f}: move {mv} not found ({path})")
            mv = str(path)
        entry = dict(nodes=keep, move=mv)
        for k in ("host", "pin", "done"):
            if e.get(k) is not None:
                entry[k] = e[k]
        if isinstance(e.get("command"), dict):
            entry["command"] = {n: e["command"][n] for n in keep}
        out.append(entry)
    return out


DONE_KEYS = ("after_s", "upright_s", "rolled_then_upright_s")


@functools.lru_cache(maxsize=None)
def _warm_knobs(ref: str) -> tuple:
    try:
        warm = moves.resolve_move(ref)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"WARM_MLP {ref!r}: {e}")
    return tuple(moves.knobs_of(ref, warm))


def config_knobs(cfg_mod) -> list[dict]:
    """The knob names a raw command may use: WARM_MLP's card's host.knobs when WARM_MLP is a library ref (alias or id)
    of a raw card, else none (slot names only)."""
    ref = getattr(cfg_mod, "WARM_MLP", None)
    return list(_warm_knobs(str(ref))) if ref and moves.is_library_ref(str(ref)) else []


def host_config(cfg_mod, nodes: list[str], knobs=()) -> dict:
    """HOST, RAW_COMMAND, COMMAND_RANGES and DONE of a config, checked and resolved:
      host           "walk-stand" | "raw"
      raw_command    raw: {node: 13 floats}, the block the in-loop TS eval and the export feed (a ranged slot that
                     RAW_COMMAND leaves unset: its range's midpoint)
      command_ranges raw: {slot index: (lo, hi)} (the env redraws those slots per training episode)
      export         raw: {node: {slot: value}}: the nonzero slots by slot name (the export's move is a path ref, and a
                     path ref takes slot names only)
      ranges         {slot: [lo, hi]}: COMMAND_RANGES by slot name (train.command_ranges; xp move card drafts knobs from it)
      done           the MoveDone dict, or None
      move_extra     the export's / TS eval's moves-entry fields: host, command, done (empty for a plain walk-stand move)
    SystemExit on anything malformed."""
    host = getattr(cfg_mod, "HOST", "walk-stand")
    if host not in ("walk-stand", "raw"):
        raise SystemExit(f"HOST {host!r}: 'walk-stand' (the default) or 'raw'")
    rc = getattr(cfg_mod, "RAW_COMMAND", None) or {}
    cr = getattr(cfg_mod, "COMMAND_RANGES", None) or {}
    if host != "raw" and rc:
        raise SystemExit("RAW_COMMAND needs HOST = 'raw'")
    if host != "raw" and cr:
        # E2.1: a walk-stand node's twist command may be redrawn per training episode (vx/vy/wz only; kit patch)
        pass
    knob_slot = {k["name"]: k["slot"] for k in knobs}
    slot_index = {s: i for i, s in enumerate(COMMAND_SLOTS)}

    def index_of(name: str) -> int:
        i = slot_index.get(knob_slot.get(name, name))
        if i is None:
            raise SystemExit(f"COMMAND_RANGES: unknown command name {name!r} (slots: {', '.join(COMMAND_SLOTS)}"
                             + (f"; knobs of WARM_MLP's card: {', '.join(knob_slot)})" if knob_slot else ")"))
        return i
    ranges: dict[int, tuple[float, float]] = {}
    for name, r in cr.items():
        lo, hi = (float(x) for x in r)
        if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
            raise SystemExit(f"COMMAND_RANGES[{name!r}] = {r!r}: want (lo, hi) with lo <= hi")
        ranges[index_of(name)] = (lo, hi)
    extra = sorted(set(rc) - set(nodes))
    if extra:
        raise SystemExit(f"RAW_COMMAND names {extra}, which are not trained nodes ({nodes})")
    raw_command, export = {}, {}
    if host == "raw":
        for n in nodes:
            v = rc.get(n, {})
            try:
                c = command_vector(v, knobs)
            except ValueError as e:
                raise SystemExit(f"RAW_COMMAND[{n!r}]: {e}")
            set_slots = set(range(13)) if not isinstance(v, dict) else {index_of(k) for k in v}
            for i, (lo, hi) in ranges.items():
                if i not in set_slots:
                    c[i] = (lo + hi) / 2
            raw_command[n] = c.tolist()
            export[n] = {COMMAND_SLOTS[i]: float(x) for i, x in enumerate(c) if x != 0}
    done = getattr(cfg_mod, "DONE", None)
    if done is not None:
        if not isinstance(done, dict) or not done or set(done) - set(DONE_KEYS) \
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 for x in done.values()):
            raise SystemExit(f"DONE {done!r}: want {{after_s?, upright_s?, rolled_then_upright_s?}}, at least one, each > 0 s")
        done = {k: float(done[k]) for k in DONE_KEYS if k in done}
    move_extra = {**({"host": "raw", "command": export} if host == "raw" else {}), **({"done": done} if done else {})}
    return dict(host=host, raw_command=raw_command, command_ranges=ranges, export=export,
                ranges={COMMAND_SLOTS[i]: list(r) for i, r in sorted(ranges.items())}, done=done, move_extra=move_extra)


def ensure_bank(cfg_mod, challenge: str, bank: str) -> Path:
    """ENV bank's path. A missing bank is recorded first when the config has BANK (make_bank.py's recipe: program,
    nodes, seeds; stock policies) into HOME (a relative bank path is under home/), in this command's own charge."""
    p = resolve(bank)
    if p.exists():
        return p
    recipe = getattr(cfg_mod, "BANK", None)
    if not recipe:
        raise SystemExit(f"ENV bank {bank}: not found; record it with make_bank.py (or give the config a BANK recipe)")
    out = Path(bank).expanduser()
    out = out if out.is_absolute() else duck_rl.HOME / out
    out.parent.mkdir(parents=True, exist_ok=True)
    nodes = recipe["nodes"] if isinstance(recipe["nodes"], str) else ",".join(recipe["nodes"])
    print(f"recording the handoff bank {out} (once: {recipe['program']} at {nodes}) ...", flush=True)
    r = bridge.run_ts("eval", ["--challenge", challenge, "--program", resolve(recipe["program"]), "--seeds",
                               recipe.get("seeds", "1000-1255"), "--bank", out, "--bank-nodes", nodes, "--quiet"])
    if r.returncode != 0 or not out.exists():
        raise SystemExit(f"recording the bank {out} failed (exit {r.returncode})")
    return out


def env_config(cfg_mod, challenge: str, compiled: dict) -> dict:
    """The ChallengeEnv cfg of a config: ENV + the top-level knobs, the trained node, the schedule, the Challenge's scene, the
    host (HOST / RAW_COMMAND / COMMAND_RANGES) and the move's done rule (DONE). Node dicts carry their `id`. A duel
    Challenge's: duel_config."""
    if "duel" in duck_rl.load_info(challenge):
        return duel_config(cfg_mod, challenge)
    scene, _ = duck_rl.ensure_scene(challenge)
    nodes = trained_nodes(cfg_mod)
    env_cfg = dict(getattr(cfg_mod, "ENV", {}), node=dict(compiled["nodes"][nodes[0]], id=nodes[0]),
                   info=duck_rl.load_info(challenge), scene=str(scene))
    prog_path = resolve(cfg_mod.PROGRAM)
    if json.loads(Path(prog_path).read_text()).get("format") == "duck-program/2":  # kit 1.15.0: the node's feed / head
        from duck_rl.duel import compile_program as compile_full
        env_cfg["program"] = compile_full(challenge, prog_path)
    for k in ("END_S", "TERM_AT_END", "FALL", "OUT_BEFORE_S"):
        if hasattr(cfg_mod, k):
            env_cfg[k.lower()] = getattr(cfg_mod, k)
    enc = config_encoder(cfg_mod)
    if enc is not None:  # kit 1.27.0: the obs gains the frozen encoder's named numbers (duck_rl/encoder.py)
        env_cfg["encoder"] = enc.name
    if getattr(cfg_mod, "RUN_PROGRAM", False):
        bad = [k for k in ("SCHEDULE", "BANK", "WORLDS", "DONE", "RAW_COMMAND", "COMMAND_RANGES")
               if getattr(cfg_mod, k, None) not in (None, [], {}, ())]
        if getattr(cfg_mod, "HOST", None) not in (None, "walk-stand"):
            bad.append(f"HOST {cfg_mod.HOST!r}")
        bad += [f"ENV {k}" for k in ("schedule", "bank", "done", "host") if env_cfg.get(k)]
        if env_cfg.get("start", "spawn") != "spawn":
            bad.append(f"ENV start {env_cfg['start']!r}")
        if bad:
            raise SystemExit(f"RUN_PROGRAM: {', '.join(bad)} not supported (the env runs the duckscript from the spawn)")
        from duck_rl.duel import compile_program as compile_full
        env_cfg.update(program=compile_full(challenge, resolve(cfg_mod.PROGRAM)), program_nodes=list(nodes),
                       base_moves=env_moves(cfg_mod, nodes))
        return env_cfg
    sched = list(getattr(cfg_mod, "SCHEDULE", []))
    if sched:
        env_cfg["schedule"] = [(float(t), dict(compiled["nodes"][nid], id=nid)) for t, nid in sched]
    hc = host_config(cfg_mod, nodes, config_knobs(cfg_mod) if getattr(cfg_mod, "HOST", None) == "raw" else ())
    if hc["host"] == "raw":
        env_cfg.update(host="raw", raw_command=hc["raw_command"], command_ranges=hc["command_ranges"])
    elif hc["command_ranges"]:
        env_cfg["command_ranges"] = hc["command_ranges"]
    if getattr(cfg_mod, "CMD_STAGE", None):
        env_cfg["cmd_stage"] = dict(cfg_mod.CMD_STAGE)
    if getattr(cfg_mod, "CMD_SCHEDULE", None):
        env_cfg["cmd_schedule"] = dict(cfg_mod.CMD_SCHEDULE)
    if getattr(cfg_mod, "VISION", False):  # the ray eye's vision moves were removed (LB-330)
        raise SystemExit("VISION was removed with the ray eye (LB-330, 2026-11-14): drop it; a move that sees is a camera "
                         "move (ENCODER = \"emb1\", configs/sprint/encoder.py)")
    if hc["done"]:
        env_cfg["done"] = hc["done"]
    elif env_cfg.get("end_on_done"):
        raise SystemExit("ENV end_on_done needs DONE (the move's done rule)")
    if env_cfg.get("bank"):
        env_cfg["bank"] = str(ensure_bank(cfg_mod, challenge, env_cfg["bank"]))
    # WORLDS (spike Challenges, 2026-10-01): a list of ts/prepare.ts --variant strings ("diameter=1.9"); the training envs
    # cycle through these worlds (env i of the pool gets WORLDS[i % n]: per-env compiled models, e.g. disc diameters).
    # One entry = that world for every env.
    worlds = list(getattr(cfg_mod, "WORLDS", []))
    if worlds:
        ws = []
        for v in worlds:
            sc, inf = duck_rl.ensure_scene(challenge, variant=v)
            ws.append(dict(scene=str(sc), info=json.loads(inf.read_text()), variant=v))
        env_cfg.update(scene=ws[0]["scene"], info=ws[0]["info"])
        if len(ws) > 1:
            env_cfg["worlds"] = ws
    return env_cfg


# the knobs a duel Challenge's config may not set in v1, and why (CONFIG_DOC)
DUEL_REFUSED = dict(
    SCHEDULE="the duckscript switches nodes by its own rules; bind the policy to each node it runs with TRAINED_NODE",
    BANK="a match starts at the spawn (no handoff bank)",
    ENV="a match starts at the spawn and its own rules end it (the ChallengeEnv cfg does not apply)",
    WORLDS="one match world", DONE="no move done rule", END_S="the match's rules end the episode",
    TERM_AT_END="the match's rules end the episode", FALL="the judge's fall rule", OUT_BEFORE_S="the judge's out rule")


def duel_config(cfg_mod, challenge: str) -> dict:
    """The DuelEnv cfg of a duel Challenge's config (duck_rl/duel_env.py): the learner's duckscript compiled in full
    (ts/prepare.ts --program-full) with TRAINED_NODE, the opponents (OPPONENT) resolved and compiled here once (the
    workers get the compiled specs), UNBOUND, and under duel.eval the in-loop TS eval's --opponent refs
    (EVAL_OPPONENTS: house refs as they are, files as absolute paths; duel.eval_refs: as the config gives them, for the
    export's train block). SystemExit on what v1 does not support."""
    from duck_rl import duel_env as de
    from duck_rl.duel import DuelSide, compile_program as compile_full
    from duck_rl.program import EXTERNAL, MatchRuntime
    for name, why in DUEL_REFUSED.items():
        v = getattr(cfg_mod, name, None)
        if v not in (None, [], {}, ()):
            raise SystemExit(f"{name} = {v!r}: not for a duel challenge ({challenge}) in this kit: {why}")
    ts = getattr(cfg_mod, "TS_EVAL", None) or {}
    for k in ("variants", "select_key"):
        if ts.get(k):
            raise SystemExit(f"TS_EVAL {k}: not for a duel challenge ({challenge}): the duel eval ranks by points")
    unbound = getattr(cfg_mod, "UNBOUND", "end")
    if unbound not in de.UNBOUND_MODES:
        raise SystemExit(f"UNBOUND = {unbound!r}: {', '.join(map(repr, de.UNBOUND_MODES))} ")
    info = duck_rl.load_info(challenge)
    scene, _ = duck_rl.ensure_scene(challenge)
    rt = MatchRuntime(info)
    nodes, program = trained_nodes(cfg_mod), resolve(cfg_mod.PROGRAM)
    # kit 1.15.0: HOST "raw" in a duel: the learner's nodes run the raw host with RAW_COMMAND (the duckscript's feed
    # replaces slots of it every program tick, as the sim does), head and body in the obs
    hc = host_config(cfg_mod, nodes, config_knobs(cfg_mod) if getattr(cfg_mod, "HOST", None) == "raw" else ())
    learner = dict(nodes=nodes, move=EXTERNAL, **({"host": "raw", "command": hc["raw_command"]} if hc["host"] == "raw" else {}))
    try:
        full = compile_full(challenge, program)
        DuelSide(challenge, rt, full, [dict(learner), *env_moves(cfg_mod, nodes)], name="learner")  # its nodes and skills run here
    except ValueError as e:
        raise SystemExit(f"PROGRAM {program} with TRAINED_NODE {nodes}: {e}")

    def side(ref: str) -> tuple[str, Path]:
        """(the TS eval's --opponent ref, the file): a house ref or stand as it is, a path absolute."""
        b = de.builtin_opponent(ref, challenge)
        if b is not None:
            return b
        f = resolve(ref)
        if not f.exists():
            raise SystemExit(f"opponent {ref!r}: no file or directory {f} (want house:easy|medium|hard, stand, a "
                             "duckscript (duck-program/1 or /2) or duck-strategy/1 file, or a submission dir)")
        return str(f), f
    opponents = []
    for ref, wt in de.parse_opponents(getattr(cfg_mod, "OPPONENT", "house:easy")):
        ts_ref, f = side(ref)
        try:
            spec = de.opponent_spec(f, challenge)
            DuelSide(challenge, rt, spec, name=ref)  # it runs in the Python match runtime (skills, signals, moves)
        except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as e:
            raise SystemExit(f"OPPONENT {ref!r} ({f}): {e}")
        opponents.append(dict(ref=ref, spec=spec, weight=wt, ts=ts_ref))
    ev = getattr(cfg_mod, "EVAL_OPPONENTS", None)
    if ev is None:
        evals = {o["ts"]: o["ref"] for o in opponents}
    else:
        evals = {}
        for r in [ev] if isinstance(ev, str) else list(ev):
            if not isinstance(r, str):
                raise SystemExit(f"EVAL_OPPONENTS entry {r!r}: a ref (house:easy|medium|hard, stand, a file or a "
                                 "submission dir); weights are OPPONENT's only (the eval plays every opponent's rounds)")
            ts_ref = side(r)[0]
            if ts_ref in evals:  # as the TS eval refuses a repeat
                raise SystemExit(f"EVAL_OPPONENTS {r!r} is given twice ({evals[ts_ref]!r} is the same opponent, {ts_ref})")
            evals[ts_ref] = r
        if not evals:
            raise SystemExit("EVAL_OPPONENTS: no opponent (leave it out for OPPONENT's)")
    duel = dict(program=full, nodes=nodes, unbound=unbound, swap=0.5, base_moves=env_moves(cfg_mod, nodes),
                opponents=[{k: o[k] for k in ("ref", "spec", "weight")} for o in opponents],
                eval=list(evals), eval_refs=list(evals.values()))
    # the raw host (kit 1.10.0, from the 2026-10-15 night; duck_rl/duel_env.py module doc): HOST / RAW_COMMAND /
    # COMMAND_RANGES as for a solo Challenge (hc above); a walk-stand config's cfg is unchanged
    if hc["host"] == "raw":
        duel.update(host="raw", raw_command=hc["raw_command"],
                    command_ranges={str(i): list(r) for i, r in hc["command_ranges"].items()})
    return dict(info=info, scene=str(scene), duel=duel)


def compile_program(challenge: str, program: Path, out: Path) -> dict:
    """The duckscript through the TS validator: node params as the sim runs them."""
    out.parent.mkdir(parents=True, exist_ok=True)
    r = bridge.run_ts("prepare", ["--challenge", challenge, "--program", program, "--compiled", out], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"prepare --program {program} failed (exit {r.returncode}): an invalid duckscript, or the TS side "
                         f"itself (`duckbench kit link .` re-links a moved CLI):\n{r.stdout}\n{r.stderr}")
    return json.loads(out.read_text())


NODE_MOVE_KEYS = ("move", "host", "command", "done")  # packages/schema strategy.ts NODE_MOVE_KEYS


def env_moves(cfg_mod, nodes: list[str]) -> list[dict]:
    """The moves the training env binds on the duckscript's untrained nodes: PROGRAM's own (node_moves, kit 1.30.0: the
    env ran those nodes' stock skills before, so a dash or a turn move trained against a slower duck than the one the
    check and the export play), then BASE_STRATEGY's (base_moves). A node has one of the two, never both (main)."""
    return [*[dict(m) for m in getattr(cfg_mod, "_FIXED_MOVES", [])], *base_moves(cfg_mod, nodes)]


def node_moves(cfg_mod, challenge: str, out: Path) -> list[dict]:
    """Kit 1.19.0: a PROGRAM whose walk/stand nodes carry their own trained move (duck-program/2 "move", host?, command?,
    done?; what `duckbench start <challenge> --move <alias>` writes) runs as the config naming those moves. ts/prepare.ts
    --node-moves binds them with the CLI's own loader (the card fills a missing host, a raw move's command from its knob
    defaults, a missing done; a raw node's feed knob names become slots) and gives the plain duckscript, written to
    <out>/program.plain.json, which becomes cfg_mod.PROGRAM. The move on the first trained node is the warm start: it sets
    WARM_MLP and, where the config leaves them unset, HOST, RAW_COMMAND (that node's command block, for every trained
    node that has none of its own) and DONE; a config that sets WARM_MLP wins (that node's move is dropped, a log line
    says so). The trained nodes run the trained policy, whatever moves they carried. Every other node's move stays bound:
    returned as duck-strategy/1 moves entries (nodes, move, pin, host?, command?, done?), the fixed moves the in-loop TS
    eval and the export bind beside the trained one. A plain PROGRAM: [] and nothing changes."""
    prog = resolve(cfg_mod.PROGRAM)
    try:
        p = json.loads(prog.read_text())
    except (OSError, ValueError):
        return []  # compile_program reports it
    ns = p.get("nodes") if isinstance(p, dict) and p.get("format") not in (moves.STRATEGY_FORMAT, "duck-rl-submission/1") else None
    if not isinstance(ns, dict) or not any(isinstance(n, dict) and any(k in n for k in NODE_MOVE_KEYS) for n in ns.values()):
        return []
    nm_path = out / "node-moves.json"
    r = bridge.run_ts("prepare", ["--challenge", challenge, "--program", prog, "--node-moves", nm_path], capture_output=True, text=True)
    if r.returncode != 0 or not nm_path.exists():
        raise SystemExit(f"PROGRAM {prog}: its node moves do not bind (exit {r.returncode}):\n{r.stdout}\n{r.stderr}")
    for line in r.stderr.splitlines():
        if line.startswith("warning:"):
            print(f"[duck-rl] PROGRAM {line}", flush=True)
    nm = json.loads(nm_path.read_text())
    plain = out / "program.plain.json"
    plain.write_text(json.dumps(nm["program"], indent=1) + "\n")
    cfg_mod.PROGRAM = str(plain)
    nodes = trained_nodes(cfg_mod)

    def part(m: dict, keep) -> dict:
        """The moves entry `m` on its nodes that `keep` accepts (a raw command block per kept node)."""
        ns_ = [n for n in m["nodes"] if keep(n)]
        e = dict(nodes=ns_, move=m["move"], pin=m["pin"], **({"host": m["host"]} if "host" in m else {}))
        if "command" in m:
            e["command"] = {n: c for n, c in m["command"].items() if n in ns_}
        if "done" in m:
            e["done"] = m["done"]
        return e
    trained = [part(m, lambda n: n in nodes) for m in nm["moves"] if any(n in nodes for n in m["nodes"])]
    fixed = [part(m, lambda n: n not in nodes) for m in nm["moves"] if any(n not in nodes for n in m["nodes"])]
    def listed(ms: list[dict]) -> str:
        return "; ".join(",".join(m["nodes"]) + " " + m["move"] for m in ms)
    print(f"[duck-rl] PROGRAM {prog} carries trained moves: {listed(nm['moves'])} -> the plain duckscript {plain}", flush=True)
    if trained:
        w = next((m for m in trained if nodes[0] in m["nodes"]), trained[0])
        others = [m for m in trained if m is not w]
        if others:
            print(f"[duck-rl] PROGRAM: trained node(s) {listed(others)} run the trained policy (started from {w['move']})", flush=True)
        if getattr(cfg_mod, "WARM_MLP", None):
            print(f"[duck-rl] PROGRAM: node {w['nodes'][0]} carries {w['move']} ({w['pin']}), but the config sets WARM_MLP "
                  f"{cfg_mod.WARM_MLP}: the config wins (warm start, HOST, RAW_COMMAND and DONE as the config sets them)", flush=True)
        else:
            cfg_mod.WARM_MLP = w["move"]
            got = [f"WARM_MLP {w['move']}"]
            if w.get("host") == "raw" and getattr(cfg_mod, "HOST", None) is None:
                cfg_mod.HOST = "raw"
                got.append("HOST raw")
            if getattr(cfg_mod, "HOST", None) == "raw" and w.get("command") and not getattr(cfg_mod, "RAW_COMMAND", None):
                first = w["command"].get(nodes[0]) or next(iter(w["command"].values()))
                cfg_mod.RAW_COMMAND = {n: dict(w["command"].get(n, first)) for n in nodes}
                got.append(f"RAW_COMMAND {cfg_mod.RAW_COMMAND}")
            if w.get("done") and getattr(cfg_mod, "DONE", None) is None:
                cfg_mod.DONE = dict(w["done"])
                got.append(f"DONE {cfg_mod.DONE}")
            print(f"[duck-rl] PROGRAM: node {w['nodes'][0]}'s move is the warm start: {', '.join(got)}", flush=True)
    if fixed:
        print(f"[duck-rl] PROGRAM: fixed moves (they run on their nodes in the training env, the TS eval and the export; "
              f"not trained): {listed(fixed)}", flush=True)
    cfg_mod._FIXED_MOVES = fixed  # env_moves: the training env binds them (kit 1.30.0)
    return fixed


def config_encoder(cfg_mod):
    """The config's ENCODER (kit 1.27.0) as a duck_rl/encoder.py Encoder, or None. Stops on an unknown name or on a
    duel Challenge (a duel learner's obs is the 61)."""
    name = getattr(cfg_mod, "ENCODER", None)
    if not name:
        return None
    from duck_rl.encoder import load
    try:
        return load(str(name))
    except ValueError as e:
        raise SystemExit(str(e))


SITE_ENCODER = None  # kit 1.27.0: the run's frozen encoder (ENCODER): every export writes the site's camera move (to_site_mlp)


def to_site_mlp(onnx: Path, path: Path) -> None:
    """A checkpoint's ONNX -> the move the site runs at `path` (tools/to_mlp.py convert): with ENCODER (SITE_ENCODER)
    the camera move over the encoder's CNN, else the plain move."""
    sys.path.insert(0, str(HERE / "tools"))
    from to_mlp import convert  # noqa: E402  (tools/to_mlp.py)
    convert(onnx, path, encoder=SITE_ENCODER)


def mlp_warm_start(path: Path):
    """lab.learn.warm_start with a duck-mlp/1 file instead of a stock skill: the weights and the frozen normaliser."""
    from duckref.mlp import DuckMlp

    def warm(actor, skill: str) -> None:
        L.load_mlp_into(actor, DuckMlp(path))  # (an encoder actor: the move's weights on its first 61 inputs)
        print(f"[duck-rl] warm start from {path}", flush=True)
    return warm


def resolve_warm_mlp(ref: str) -> dict:
    """WARM_MLP -> {"id", "json", "bin", "alias"}: a move id or alias through the kit's move index (bridge.move_path),
    a path like any config path (the kit, the repo, HOME)."""
    if moves.is_library_ref(ref):
        try:
            return moves.resolve_move(ref)
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            raise SystemExit(f"WARM_MLP {ref!r}: {e}")
    return moves.resolve_path(resolve(ref))


def parent_encoder_problem(init: Path, flag: str, encoder) -> str | None:
    """PR #1 R12-1: an --init / --resume parent must have learnt on the encoder the config reads. Its weights' layer 0
    is fitted to that encoder's numbers, so another encoder's numbers would train (and export as the config's camera
    move) a move that never saw them. The parent's own camera move (policy.mlp.json beside it, or the checkpoint's
    sibling .mlp.json) must not read a retired picture (duck_rl/encoder.py RETIRED_PICTURES, the site's words), and the
    encoder its training recorded (kit_files.json in its run dir: beside it, or home/runs/<label>/ of a submission) must be
    the config's ENCODER (`encoder`: an Encoder or None; a blind training records none). The problem, or None."""
    from duck_rl.encoder import PICTURE, RETIRED_PICTURES  # noqa: PLC0415
    init = init.resolve()
    for mlp in (init.parent / "policy.mlp.json", init.with_suffix(".mlp.json")):
        with contextlib.suppress(ValueError, OSError):
            cam = json.loads(mlp.read_text()).get("camera") if mlp.is_file() else None
            pic = cam.get("picture") if isinstance(cam, dict) else None
            if pic in RETIRED_PICTURES:
                return f'--{flag} {init}: its move reads picture "{pic}", which is retired: {RETIRED_PICTURES[pic]}'
    kf = init.parent / "kit_files.json"
    if not kf.is_file():
        with contextlib.suppress(ValueError, OSError):
            sub = json.loads((init.parent / "submission.json").read_text()).get("train") or {}
            run = duck_rl.RUNS / str(sub.get("label") or "")
            if sub.get("label") and (run / "kit_files.json").is_file():
                doc = json.loads((run / "kit_files.json").read_text())
                kf = run / "kit_files.json" if doc.get("session") == sub.get("session") else kf
    try:
        doc = json.loads(kf.read_text()) if kf.is_file() else {}
    except (ValueError, OSError):
        doc = {}
    if "encoder" not in doc:
        return None  # no record (a kit before 1.27.0, or none found): only the picture check above applies
    rec = doc["encoder"] if isinstance(doc["encoder"], dict) else {}
    pid, cid = rec.get("id"), (encoder.id if encoder is not None else None)
    if pid == cid:
        return None
    was = f"encoder {rec.get('name')} ({pid})" if pid else "no encoder (a blind move)"
    now = f"encoder {encoder.name} ({cid})" if encoder is not None else "no encoder (ENCODER is not set)"
    retrain = f'retrain the move on "{PICTURE}" (a camera encoder for palette v2)'
    keep = ""  # PR #1 R19-2: every kit encoder reads the same picture, so naming the parent's encoder is the direct fix
    if pid and rec.get("name"):
        from duck_rl.encoder import load  # noqa: PLC0415
        with contextlib.suppress(ValueError, OSError):
            load(str(rec["name"]))
            keep = (f' To keep training this move, set ENCODER = "{rec["name"]}" in the config (both encoders read the '
                    f"same picture).")
    return (f"--{flag} {init}: its weights learnt on {was} ({kf}); this config reads {now}. A move's layer 0 fits the "
            f"encoder it learnt on.{keep} Otherwise {retrain} from a parent trained on the config's encoder, or start "
            f"without --{flag}.")


def init_parent(init: Path) -> dict:
    """The --init checkpoint as a lineage parent: the move id of its weights (converted as an export converts them: the
    sibling .onnx of a ckpt-N.pt, the run's policy.onnx for its ckpt.pt, else the .pt itself), its steps, and the train
    lineage beside it (the run's stats.json, or the submission whose ckpt.pt it is); `minutes` = the recorded
    train.minutes of the kit's own run it continues (submissions/<label>/ or home/runs/<label>/), else 0."""
    sys.path.insert(0, str(HERE / "tools"))
    from to_mlp import convert  # noqa: E402
    init = init.resolve()
    sub = init.parent / "submission.json"
    sub_meta = json.loads(sub.read_text()) if sub.is_file() else {}
    train_meta = sub_meta.get("train") if isinstance(sub_meta.get("train"), dict) else {}
    ck = torch.load(init, map_location="cpu")
    if train_meta.get("ckpt_sha256") == moves.sha256_file(init) and (init.parent / "policy.mlp.json").exists():
        mid, lin = moves.move_id_of(init.parent / "policy.mlp.json"), train_meta.get("lineage")  # a submission's ckpt.pt
    else:
        src = next((c for c in (init.with_suffix(".onnx"), init.parent / "policy.onnx" if init.name == "ckpt.pt" else None)
                    if c is not None and c.exists()), init)
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            convert(src, Path(td) / "p.mlp.json", encoder=SITE_ENCODER)
            mid = moves.move_id_of(Path(td) / "p.mlp.json")
        stats = init.parent / "stats.json"
        lin = train_meta.get("lineage")  # a submission's last.pt (kit 1.8.1): the same run's lineage
        if stats.is_file():
            with contextlib.suppress(ValueError):
                lin = json.loads(stats.read_text()).get("lineage")
    minutes = 0.0
    if init.parent.parent == (HERE / "submissions").resolve():
        minutes = train_meta.get("minutes") or 0.0
    elif init.parent.parent == duck_rl.RUNS.resolve() and (init.parent / "stats.json").is_file():
        with contextlib.suppress(ValueError):
            minutes = json.loads((init.parent / "stats.json").read_text()).get("minutes") or 0.0
    ck = ck if isinstance(ck, dict) else {}
    # steps: its session's (the lineage's this_step counts them); run_steps: the run's (kit 1.15.0 checkpoint names)
    return dict(id=mid, path=str(init), steps=ck.get("session_steps", ck.get("steps")), run_steps=ck.get("steps"),
                lineage=lin, minutes=float(minutes))


STEP_OFFSET = 0  # kit 1.15.0: a --resume session's checkpoints continue the run's step count (ckpt-<run steps>)


def ckpt_steps(ck: Path) -> int:
    """This session's env steps of a periodic checkpoint ckpt-<run steps>.onnx/.pt (the names carry the run's count)."""
    return int(ck.stem.split("-")[1]) - STEP_OFFSET


def ckpt_of(onnx: Path, run: Path, steps: int) -> Path | None:
    """The checkpoint .pt of an exported onnx: ckpt-N.onnx -> ckpt-N.pt; the run's final policy.onnx -> the final
    checkpoint ckpt-<steps>.pt (actor with its std, critic), else the lab's plain ckpt.pt beside it."""
    if onnx.name == "policy.onnx":
        return next((c for c in (run / f"ckpt-{STEP_OFFSET + steps:07d}.pt", run / "ckpt.pt") if c.exists()), None)
    c = onnx.with_suffix(".pt")
    return c if c.exists() else None


def checkpoints(run: Path) -> list[Path]:
    """The run's periodic ckpt-<steps>.onnx, oldest first. By the step count, not the name: past 9,999,999 steps the
    names get 8 digits, and as text ckpt-10000000 sorts before ckpt-9750000."""
    return sorted(run.glob("ckpt-*.onnx"), key=lambda p: int(p.stem.split("-")[1]))


def prior_best(init: Path, sessions: int) -> dict | None:
    """kit 1.15.0: the best scored checkpoint of the earlier sessions of a --resume run, so the export picks the best of
    the whole run: {rec (its in-loop eval record), session, ckpt (its checkpoint name), run_steps, lineage, files
    {name: bytes}: its weights (.mlp.json + .mlp.bin, or .onnx) and its checkpoint .pt, the weights first}. From a submission (train.run_best, else the
    record of the checkpoint it exported: a kit < 1.15.0 export is its session's best) with its policy.mlp.json and
    ckpt.pt; from a run dir (stats.json run_best, else its best record) with the files it names. Read into memory
    before the run dir of the same label moves aside. None when unknown or its files are gone."""
    d = init.resolve().parent
    for f, key in ((d / "submission.json", "train"), (d / "stats.json", None)):
        if not f.is_file():
            continue
        try:
            meta = json.loads(f.read_text())
        except ValueError:
            return None
        meta = meta.get(key) if key else meta
        if not isinstance(meta, dict):
            return None
        rb = meta.get("run_best") if isinstance(meta.get("run_best"), dict) else None
        if rb is None:
            recs = [r for r in meta.get("ts_evals") or [] if r.get("rank_key") is not None]
            rec = (next((r for r in recs if Path(r["onnx"]).name == Path(str(meta.get("onnx"))).name), None) if key
                   else max(recs, key=lambda r: tuple(r["rank_key"]), default=None))
            if rec is None:
                return None
            off = ((meta.get("curve") or {}).get("offset") or {}).get("steps") or 0
            rb = dict(rec=rec, session=sessions, ckpt=Path(rec["onnx"]).stem, run_steps=int(off) + int(rec["steps"]),
                      lineage=meta.get("lineage"), files=[Path(rec["onnx"]).name, Path(rec["onnx"]).with_suffix(".pt").name])
        names = ["policy.mlp.json", "ckpt.pt", "policy.mlp.bin"] if key else rb.get("files") or []
        if len(names) < 2 or not all((d / n).is_file() for n in names):
            print(f"[duck-rl] whole-run best: the files of {rb.get('ckpt')} are not in {d}: the export picks from this "
                  "session", flush=True)
            return None
        ext = lambda n: next(e for e in (".mlp.json", ".mlp.bin", ".onnx", ".pt") if n.endswith(e))  # noqa: E731
        return dict(rb, files={f"{rb['ckpt']}{ext(n)}": (d / n).read_bytes() for n in names})  # (named by its checkpoint)
    return None


def same_eval(a: dict, b: dict) -> bool:
    """Two in-loop eval records on the same seeds (and opponents): their rank keys compare."""
    def seeds(r):
        return sorted(json.dumps([p[0]] + p[2:]) for p in r.get("per_seed") or [])
    return (a.get("kind") == b.get("kind") and a.get("n") == b.get("n") and len(a["rank_key"]) == len(b["rank_key"])
            and (not a.get("per_seed") or not b.get("per_seed") or seeds(a) == seeds(b)))


def reward_fn(cfg_mod, params: dict, names: list[str] | None = None):
    def terms(s: dict) -> dict:
        r = cfg_mod.reward(s, params)
        n = s["t"].shape[0]
        out = {k: np.broadcast_to(np.asarray(v, np.float64), (n,)) for k, v in r.items()}
        for k, v in out.items():
            if not np.all(np.isfinite(v)):
                raise FloatingPointError(f"reward term {k!r} is not finite")
        if names is not None:
            for k in names:
                out.setdefault(k, np.zeros(n))
        return out
    return terms


def term_names(cfg_mod, params: dict, env_cfg: dict) -> tuple[list[str], float]:
    """The reward's term names (from 3 env steps) and the env's episode end (s)."""
    env = make_env(env_cfg, 12345)
    env.reset()
    S = []
    for _ in range(3):
        _, _, st, term, to = env.step(np.zeros(14, np.float32))
        S.append(st)
        if term or to:
            env.reset()
    return sorted(reward_fn(cfg_mod, params)(make_unpack(env_cfg["info"])(np.stack(S)))), env.end_s


SPS_PER_WORKER_LOW = 1000  # env steps/s per worker below which the machine is busy (a quiet one: ~2.5-3.5k per worker)


def load_note(workers: int) -> None:
    """Before training (kit 1.15.0): a warning when the machine is already busy (the 1-min load average over the cores),
    so a slow run is not mistaken for a slow config."""
    import os
    try:
        load, cores = os.getloadavg()[0], os.cpu_count() or 1
    except (AttributeError, OSError):
        return
    if load > 0.6 * cores:
        print(f"[duck-rl] WARNING: this machine is busy (load {load:.1f} on {cores} cores) before training starts: expect "
              f"fewer steps/s than usual and a later best checkpoint; close other heavy work, or train longer", flush=True)


def slow_machine_note(sps: int, workers: int) -> None:
    """At iteration 10 (kit 1.15.0): a warning when the measured steps/s are far below a quiet machine's."""
    if workers and sps < SPS_PER_WORKER_LOW * workers:
        print(f"[duck-rl] WARNING: {sps} steps/s with {workers} workers ({sps // max(workers, 1)} per worker; a quiet "
              f"machine does ~2.5-3.5k per worker): the machine is loaded or throttled, so this run will see fewer steps "
              f"than its minutes suggest", flush=True)


HOUSE_TITLE = "Arena bot"


def write_program(program: Path, out: Path, label: str) -> None:
    """The export's program.json (kit 1.15.0): the duckscript with the player's own title. A copied Arena bot's title
    ("Arena bot: medium") never ships as the player's (the duck would show as a house bot): it becomes
    "<label> (trained from Arena bot: medium's duckscript)", at most 80 characters; a title of the player's own stays."""
    p = json.loads(Path(program).read_text())
    meta = dict(p.get("meta") or {})
    title = str(meta.get("title") or "").strip()
    if not title or title.lower().startswith(HOUSE_TITLE.lower()):
        new = f"{label} (trained from {title}'s duckscript)" if title else f"{label} (trained)"
        meta["title"] = new if len(new) <= 80 else f"{label} (trained)"[:80]
        p["meta"] = meta
    out.write_text(json.dumps(p, indent=1) + "\n")


def portable_paths(v, out: Path):
    """A submission's train block without this machine's paths (kit 1.15.0; the export is meant for upload): a file in
    the submission folder by its name, any other absolute path as <local>/<its last two parts> (a run dir, a checkpoint,
    a warm-start file), so no home folder or user name reaches the server."""
    import os
    home = str(Path.home())

    def fix(x):
        if isinstance(x, dict):
            return {k: fix(y) for k, y in x.items()}
        if isinstance(x, list):
            return [fix(y) for y in x]
        if isinstance(x, str) and (os.path.isabs(x) or x.startswith("~") or home in x):
            p = Path(x.replace("~", home, 1) if x.startswith("~") else x)
            try:
                return str(p.relative_to(out))
            except ValueError:
                return "<local>/" + "/".join(p.parts[-2:])
        return x
    return fix(v)


def export_base_moves(base: list[dict], out: Path) -> list[dict]:
    """BASE_STRATEGY's moves into the export: a file move copied beside policy.mlp.json as base-<i>.mlp.json + .bin (so the
    submission folder is whole: `duckbench submit` uploads it as it is), pinned to its id; a library ref kept as it is."""
    entries = []
    for i, e in enumerate(base):
        e = dict(e)
        if not moves.is_library_ref(str(e["move"])):
            src = Path(e["move"])
            dst = out / f"base-{i}.mlp.json"
            shutil.copy(src, dst)
            shutil.copy(moves.bin_of(src), moves.bin_of(dst))
            e["move"], e["pin"] = dst.name, moves.move_id_of(dst)
        entries.append(e)
    return entries


def export_submission(label: str, info: dict, program: Path, nodes: list[str], onnx: Path, out: Path, meta: dict,
                      challenge: str | None = None, ckpt: Path | None = None, world: str | None = None,
                      move_extra: dict | None = None, warm: dict | None = None, last: Path | None = None,
                      fixed_moves: list[dict] | None = None, base: list[dict] | None = None) -> Path:
    """submissions/<label>/: program.json, policy.mlp.json + .bin (from `onnx`), ckpt.pt (a copy of `ckpt`, the
    checkpoint of those weights; train.ckpt_sha256) and submission.json: duck-strategy/1 with the move pinned to its id
    (moves[0].pin; `world` = the world hash the TS eval reported, if any; `move_extra` = host_config's host, command
    and done fields of the move entry), or, for a spike Challenge (version 0), the legacy duck-rl-submission/1 (train.pin;
    no raw host or done there). `warm` = {"move": library ref, "pin": its id} of the library move the run started
    from: warm_start.json is then the same duck-strategy/1 with that move and no train block, to score where the run
    started on the same seeds; without one (a stock start, kit 1.11.1) warm_start.json is the duckscript itself. `last` = the run's newest checkpoint (kit 1.8.1): last.pt, plus its MLX Adam sidecar as
    opt.pt (lab.learn.opt_sidecar of last.pt), so `train.py --resume submissions/<label>/last.pt` continues the run
    where it stopped (weights, std, critic, optimizer state, LR); ckpt.pt (the exported weights) finds opt.pt only
    when it is that same checkpoint. `fixed_moves` (kit 1.19.0, node_moves): the moves PROGRAM's other nodes carry,
    entries after the trained move's (in warm_start.json too). `base` (kit 1.17.0, base_moves): BASE_STRATEGY's moves,
    entries after the fixed ones (export_base_moves)."""
    sys.path.insert(0, str(HERE / "tools"))
    from to_mlp import convert  # noqa: E402  (tools/to_mlp.py)
    fixed = [dict(m) for m in fixed_moves or []]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    write_program(program, out / "program.json", label)
    for m in fixed:  # a fixed move given by path travels with the export, beside program.json (its id pins it)
        if not moves.is_library_ref(m["move"]):
            src = Path(m["move"])
            shutil.copy(src, out / f"{m['pin']}.mlp.json")
            shutil.copy(src.with_name(src.name[: -len(".json")] + ".bin"), out / f"{m['pin']}.mlp.bin")
            m["move"] = f"{m['pin']}.mlp.json"
    if onnx.name.endswith(".mlp.json"):  # (kit 1.15.0: an earlier session's export, the whole run's best)
        shutil.copy(onnx, out / "policy.mlp.json")
        shutil.copy(onnx.with_name(onnx.name[: -len(".json")] + ".bin"), out / "policy.mlp.bin")
    else:
        to_site_mlp(onnx, out / "policy.mlp.json")
    pin = moves.move_id_of(out / "policy.mlp.json")
    train = dict(meta)
    if ckpt is not None:
        shutil.copy(ckpt, out / "ckpt.pt")
        train.update(ckpt_sha256=moves.sha256_file(out / "ckpt.pt"), ckpt_from=ckpt.name)
    if last is not None:
        shutil.copy(last, out / "last.pt")
        side = L.opt_sidecar(last)
        if side.exists():
            shutil.copy(side, L.opt_sidecar(out / "last.pt"))
        train.update(last_from=last.name, last_opt=side.exists())
    # the config's Challenge id (a spike alias such as saucer-peak scores with its own metric), else the info's
    did, version = challenge or info["challenge"]["id"], info["challenge"]["version"]
    if version >= 1:
        sub = {"format": moves.STRATEGY_FORMAT, "challenge": {"id": did, "version": version, **({"world": world} if world else {})},
               "program": "program.json", "moves": [{"nodes": nodes, "move": "policy.mlp.json", "pin": pin, **(move_extra or {})}, *fixed],
               "notes": f"train.py {label}", "train": train}
    else:  # a spike Challenge (ts/spike_challenges.ts, version 0): duck-strategy/1 wants a Challenge version >= 1
        if move_extra or fixed:
            raise SystemExit(f"spike challenge {did} (version 0): the legacy export cannot carry the raw host, DONE or a node's own move")
        sub = {"format": "duck-rl-submission/1", "challenge": {"id": did, "version": version}, "program": "program.json",
               "policies": [{"nodes": nodes, "mlp": "policy.mlp.json"}], "notes": f"train.py {label}", "train": dict(train, pin=pin)}
    base_entries = export_base_moves(base or [], out)
    if base_entries:
        if version < 1:
            raise SystemExit(f"spike challenge {did} (version 0): BASE_STRATEGY needs a duck-strategy/1 export")
        sub["moves"] += base_entries
    sub = portable_paths(sub, out)
    (out / "submission.json").write_text(json.dumps(sub, indent=1))
    if warm is not None and version >= 1:
        ws = {k: v for k, v in sub.items() if k != "train"}
        ws.update(moves=[dict(sub["moves"][0], **warm), *sub["moves"][1:]], notes=f"train.py {label}: its warm start {warm['move']}")
        (out / "warm_start.json").write_text(json.dumps(ws, indent=1))
        print(f"[duck-rl] warm_start.json: the same strategy with {warm['move']} (score both on the same seeds)", flush=True)
    elif version >= 1 and (fixed or base_entries):
        # kit 1.19.0 / 1.17.0: the trained nodes' stock skills, the other nodes' moves (the duckscript's fixed moves, then
        # BASE_STRATEGY's)
        what = " and ".join(w for w, on in (("the duckscript's fixed moves", fixed), ("BASE_STRATEGY's moves", base_entries)) if on)
        ws = {k: v for k, v in sub.items() if k != "train"}
        ws.update(moves=sub["moves"][1:], notes=f"train.py {label}: its stock start with {what}")
        (out / "warm_start.json").write_text(json.dumps(ws, indent=1))
        print(f"[duck-rl] warm_start.json: the duckscript with its stock skills and {what}, where the run started", flush=True)
    elif version >= 1:  # kit 1.11.1: a run from a stock skill starts from the duckscript alone (its nodes run the stock skills)
        shutil.copy(program, out / "warm_start.json")
        print("[duck-rl] warm_start.json: the duckscript with its stock skills, where the run started (score both on the same seeds)", flush=True)
    print(f"[duck-rl] exported {sub['format']} {out}: move {pin}" + (f", ckpt.pt = {ckpt.name}" if ckpt else ", no checkpoint .pt found")
          + (f", last.pt = {last.name}" + (" + opt.pt" if train.get("last_opt") else "") if last is not None else ""), flush=True)
    return out


TS_EVAL_DEFAULTS = dict(
    enabled=True,
    seeds="100-115",  # any public seeds; default disjoint from the 0-31 dev check (the official score is on hidden seeds)
    every_s=30.0,     # at most one eval per this many seconds; each takes the newest checkpoint not yet scored
    nice=10,          # the eval process's niceness (it overlaps training on one core)
    select="best",    # the submission: "best" (the Challenge's rank key: the mean of the best half of the seeds' natural
                      # results, kit 1.14.0; a crowd / duel Challenge: the mean points; a tie keeps the earlier checkpoint) or "last"
    final=True,       # also score the final weights when the run ends (time permitting)
    variants=None,    # spike Challenges (ts/spike_challenges.ts): a list of eval.ts --variant strings, e.g. ["mass=2.5", "mass=9"]; each
                      # is scored on `seeds` and the record's result / mean / worst / rank key are the MEAN over the variants
    select_key=None,  # a per-seed Challenge diagnostic (e.g. the saucer's "M3", "treadmill", "fling_dist"): the record's rank key
                      # becomes [its median over the seeds (mean over variants), its worst seed]; larger is better
    save_runs="improved",  # the evals of a real Challenge (version >= 1, no variants) as duck-run/1 runs in the local run store
                      # (eval.ts --save-run; the viewer `duckbench view`): "improved" keeps the runs of checkpoints that
                      # beat the session's best rank key so far (the first always does) plus the exported one; "all" every
                      # eval's (~150 KB per 30-s seed: ~2.4 MB per 16-seed eval; a duel Challenge's run holds 8 rounds of
                      # its eval, ~1.7 MB); "off" none (True = "all", False = "off"); --save-runs / --no-save-runs
)
TS_EVAL_DUEL = dict(every_s=60.0)  # a duel Challenge's defaults over TS_EVAL_DEFAULTS (an eval plays 2 sides x each opponent)


def save_runs_mode(v) -> str:
    """TS_EVAL save_runs -> "improved" | "all" | "off" (old configs: True = "all", False = "off")."""
    if v is True or v is False:
        return "all" if v else "off"
    if v not in SAVE_RUNS_MODES:
        raise SystemExit(f"TS_EVAL save_runs = {v!r}: one of {', '.join(map(repr, SAVE_RUNS_MODES))}")
    return v


def writable_dir(d: Path) -> bool:
    """d exists (made if missing) and this process can create files in it."""
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(d, os.W_OK | os.X_OK)


def player_home() -> Path:
    """The CLI's data folder, as apps/cli/src/home.ts duckBenchHome() resolves it: DUCKBENCH_HOME, else ~/.duckbench-player
    in the player package (standalone kit), else ~/.duckbench (the repo's CLI)."""
    if os.environ.get("DUCKBENCH_HOME"):
        return Path(os.environ["DUCKBENCH_HOME"]).expanduser()
    return Path.home() / (".duckbench-player" if bridge.STANDALONE else ".duckbench")


def saved_agent_model() -> str | None:
    """The model the CLI remembered from an earlier shell (agent.json in player_home(), apps/cli/src/agentMemory.ts, A8);
    None when there is none or it is not a valid train.model."""
    try:
        d = json.loads((player_home() / "agent.json").read_text())
    except (OSError, ValueError):
        return None
    m = d.get("model") if isinstance(d, dict) else None
    m = m.strip() if isinstance(m, str) else ""
    return m if MODEL_RE.fullmatch(m) else None


def runs_dir() -> Path:
    """The CLI's run store, as apps/cli/src/runs.ts runsDir() resolves it: DUCKBENCH_RUNS_DIR, else
    $DUCKBENCH_HOME/runs (~/.duckbench-player/runs, LB-133) in the player package (standalone kit), else <repo>/data/local/runs. A
    repo whose run store can't be written (an experiment's pinned repo, read-only: setup.py) gets HOME/checkpoint-runs
    instead, with a one-line note (kit 1.8.2; `DUCKBENCH_RUNS_DIR=<it> duckbench view` replays them)."""
    if os.environ.get("DUCKBENCH_RUNS_DIR"):
        return Path(os.environ["DUCKBENCH_RUNS_DIR"]).expanduser().resolve()
    if bridge.STANDALONE:
        return (Path(os.environ.get("DUCKBENCH_HOME") or Path.home() / ".duckbench-player") / "runs").resolve()
    store = (bridge.REPO / "data" / "local" / "runs").resolve()
    if writable_dir(store):
        return store
    alt = (duck_rl.HOME / "checkpoint-runs").resolve()
    print(f"[duck-rl] checkpoint runs: {store} is read-only (an experiment's pinned repo), so they go to {alt}; "
          f"replay them with DUCKBENCH_RUNS_DIR={alt} duckbench view", flush=True)
    return alt


def dir_bytes(d: Path) -> int:
    """The bytes of the files under d (a run dir)."""
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


def mb(n: float) -> str:
    """Bytes as MB for the log: 0.62, 2.4, 17, 290."""
    x = n / 1e6
    return f"{x:.2f}" if x < 1 else f"{x:.1f}" if x < 10 else f"{x:.0f}"


def deadline_left(deadline: float | None) -> float:
    return 1e9 if deadline is None else deadline - time.time()


class TsEval:
    """In-loop TS eval: while PPO trains, a nice'd background process scores the newest checkpoint (ckpt-*.onnx, written
    every --ckpt-every env steps) in the TS sim (the official eval code, ts/eval.ts, on cfg seeds) with the full
    program. One eval at a time; results -> <run>/ts_eval.jsonl and the log; the best by the Challenge's rank key (the
    summary's rank_key, kit 1.14.0, Emil 2026-10-30: [the mean of the best half of the seeds' natural results] in the
    Challenge's direction, so [-result time] on a race Challenge and [result rpm] on a score Challenge; larger is better;
    a tie keeps the earlier checkpoint; the plain mean and the worst seed are logged beside it) is kept for the
    submission. With cfg save_runs ("improved" / "all") and `run_meta` (a real Challenge, no variants), each eval also writes
    a duck-run/1 run (eval.ts --save-run --run-meta: the `train` block with this session, the stage, steps and training
    seconds; the eval process writes it, so training never waits on it) into a staging dir, <runs>/.staging/<session>
    (the same disk; `duckbench runs` lists only run-id dirs), and poll() moves it into the run store with one rename if
    it is kept: "all" keeps every run; "improved" those whose rank key beats the best so far (strictly, element by
    element; the first eval always does). The most recent eval's run that was not kept is held in staging until the
    next eval replaces it, so mark_final() can still keep the exported checkpoint's run (select="last") without a
    re-run; close() deletes the staging dir. The eval runs in the trainer's process group, so the budget runner's stop
    covers it. A duel Challenge (`opponents`: eval.ts --opponent refs): matches, both sides of every seed against each
    opponent; the rank key is [mean points per round, the worst seed's], each record carries a `duel` summary (W / D /
    L, the BT estimate, per opponent) and per_seed rows [seed, points, opponent]; its saved run is a match run (the
    CLI's format) of a viewable subset, the first opponent's rounds on the first 4 eval seeds from both sides
    (ts/duel_eval.ts RUN_SEEDS), with the train block of the whole eval."""

    def __init__(self, cfg: dict, out: Path, challenge: str, program: Path, nodes: list[str], move_extra: dict | None = None,
                 run_meta: dict | None = None, ckpt_every: int | None = None, opponents: list[str] | None = None,
                 fixed_moves: list[dict] | None = None, base: list[dict] | None = None):
        self.cfg, self.out, self.challenge, self.program, self.nodes = cfg, out, challenge, program, nodes
        self.base_moves = list(base or [])  # BASE_STRATEGY's entries (base_moves, kit 1.17.0)
        # kit 1.19.0: the moves PROGRAM's other nodes carry (node_moves), then BASE_STRATEGY's, bound beside the trained
        # one in every eval (the start's eval too)
        self.fixed = [x for m in [*(fixed_moves or []), *self.base_moves] for x in ("--move", json.dumps(m))]
        self.opponents = list(opponents or [])  # a duel Challenge's eval.ts --opponent refs
        self.move_extra = move_extra or {}  # the raw host's command blocks and the done rule (host_config)
        self.save_mode = save_runs_mode(cfg.get("save_runs", "improved"))
        # the run store's `train` block minus stage/steps/elapsed_s (set per eval); None: no runs saved
        self.run_meta = run_meta if self.save_mode != "off" and not (cfg.get("variants") or []) else None
        self.runs_dir = runs_dir() if self.run_meta is not None else None
        self.staging = self.runs_dir / ".staging" / run_meta["session"] if self.run_meta is not None else None
        self.ckpt_every = ckpt_every
        self.held = None  # the most recent eval's record whose run was not kept (its run still in staging)
        self.run_bytes: list[int] = []  # every saved eval's run size (kept or not): what "all" would have used
        self.kept_bytes = 0
        self.kept = 0
        self.train_minutes: float | None = None  # this run's training minutes (the estimate's total)
        self.dir = out / "ts_eval"
        self.dir.mkdir(exist_ok=True)
        self.log = open(out / "ts_eval.jsonl", "a")
        self.proc, self.cur, self.done, self.records = None, None, set(), []
        self.t_last = 0.0
        self.stage = 0  # the next eval's 0-based ordinal in this session (the saved run's train.stage)
        self.progress: list[tuple[int, float]] = []  # (env steps, training seconds) per training iteration: a checkpoint's elapsed_s
        self.cycles: list[tuple[int, int]] = []  # (env steps, LR_ANNEAL "cos3" cycle) per training iteration
        self.train_end = None  # epoch the training stops: no new in-loop eval that would still run then (the final one needs the time)
        self.start_rec: dict | None = None  # N26: the start's eval (start_eval), on the same seeds and opponents
        self.busy = None  # N25: a duck_rl.busy.BusyWatch fed every finished eval (run() sets it)

    def elapsed_of(self, steps: int) -> float:
        """Training seconds when `steps` env steps were done (the first iteration that reached them; else the last known)."""
        return next((e for s_, e in self.progress if s_ >= steps), self.progress[-1][1] if self.progress else 0.0)

    def cycle_of(self, steps: int) -> int | None:
        """LR_ANNEAL "cos3": the LR cycle of the iteration that trained up to `steps` env steps (None: no cycles)."""
        return next((c for s_, c in self.cycles if s_ >= steps), self.cycles[-1][1] if self.cycles else None)

    def start(self, onnx: Path, steps: int, force: bool = False, elapsed_s: float | None = None) -> bool:
        if self.proc is not None or onnx in self.done or not onnx.exists():
            return False
        if not force and time.time() - self.t_last < float(self.cfg["every_s"]):
            return False
        est = max((r["eval_s"] for r in self.records), default=10.0)
        if not force and self.cfg["final"] and self.train_end is not None and time.time() + est > self.train_end:
            return False
        sys.path.insert(0, str(HERE / "tools"))
        from to_mlp import convert  # noqa: E402
        d = self.dir / onnx.stem
        d.mkdir(exist_ok=True)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            to_site_mlp(onnx, d / "policy.mlp.json")
        mlp = d / "policy.mlp.json"
        bind = (["--move", json.dumps(dict(nodes=self.nodes, move=str(mlp), **self.move_extra))] if self.move_extra
                else ["--policy", f"{','.join(self.nodes)}={mlp}"])
        stage = self.stage
        save = []
        el = float(elapsed_s if elapsed_s is not None else self.elapsed_of(steps))  # the checkpoint's training seconds
        if self.run_meta is not None:
            train = dict(self.run_meta, stage=stage, steps=int(steps), elapsed_s=round(el, 1))
            self.staging.mkdir(parents=True, exist_ok=True)
            save = ["--save-run", "--run-meta", json.dumps({"train": train}), "--runs-dir", str(self.staging)]
        cmd = self.eval_cmd(bind, save, d)
        variants = self.cfg.get("variants") or []
        if variants:  # one eval.ts per variant, in sequence, in one niced shell (the trainer's process group)
            import shlex
            cmds = [" ".join(shlex.quote(x) for x in cmd[:-3] + ["--variant", v, "--out", str(d / f"summary-{i}.json"), "--quiet"])
                    for i, v in enumerate(variants)]
            cmd = ["sh", "-c", " && ".join(cmds)]
        self.proc = subprocess.Popen(cmd, cwd=bridge.cwd(), stdout=subprocess.DEVNULL, stderr=open(d / "stderr.txt", "w"))
        self.cur = dict(onnx=str(onnx), dir=str(d), steps=steps, t0=time.time(), stage=stage, elapsed_s=el, lr_cycle=self.cycle_of(steps))
        self.stage += 1
        self.done.add(onnx)
        self.t_last = time.time()
        return True

    def eval_cmd(self, bind: list[str], save: list[str], d: Path) -> list[str]:
        """The niced eval.ts command line: the program with `bind` (the trained nodes' weights, else nothing: their stock
        skills) and the fixed moves, the opponents, the eval seeds; its summary to <d>/summary.json."""
        opp = [x for o in self.opponents for x in ("--opponent", o)]
        return ["nice", "-n", str(int(self.cfg["nice"])), *bridge.ts_argv("eval"), "--challenge", self.challenge,
                "--program", str(self.program), *bind, *self.fixed, *opp, "--seeds", str(self.cfg["seeds"]), *save, "--out",
                str(d / "summary.json"), "--quiet"]

    def start_eval(self, mlp: Path | None, what: str) -> bool:
        """N26 (newbie test 2026-11-02): the start's eval, once, before the first checkpoint, on the same seeds,
        opponents and program as every checkpoint's, so the end's start -> trained comparison (start_vs_trained) needs
        no re-run by the agent. `mlp`: the start's weights (a duck-mlp/1 file: WARM_MLP's), bound as a checkpoint's;
        None: the trained nodes run their stock skills (warm_start.json of a stock start). Not with variants or a
        select_key (their records are not the Challenge's result). Never saved as a run, never in ts_eval.jsonl or the
        pick of the best checkpoint; the record goes to <run>/start_eval.json."""
        if self.proc is not None or self.cfg.get("variants") or self.cfg.get("select_key"):
            return False
        if mlp is None:
            bind = []
        elif self.move_extra:
            bind = ["--move", json.dumps(dict(nodes=self.nodes, move=str(mlp), **self.move_extra))]
        else:
            bind = ["--policy", f"{','.join(self.nodes)}={mlp}"]
        d = self.dir / "start"
        d.mkdir(exist_ok=True)
        self.proc = subprocess.Popen(self.eval_cmd(bind, [], d), cwd=bridge.cwd(), stdout=subprocess.DEVNULL,
                                     stderr=open(d / "stderr.txt", "w"))
        self.cur = dict(onnx=None, dir=str(d), start=what, steps=0, t0=time.time(), stage=None, elapsed_s=None, lr_cycle=None)
        return True

    def poll(self) -> None:
        if self.proc is None or self.proc.poll() is None:
            return
        rc, cur, self.proc = self.proc.returncode, self.cur, None
        f = Path(cur["dir"]) / "summary.json"
        name = f"the start ({cur['start']})" if cur.get("start") else Path(cur["onnx"]).name
        variants = self.cfg.get("variants") or []
        if variants and rc == 0:  # combine: the mean over the variants of the mean, the worst seed, the rank key
            parts = [json.loads((f.parent / f"summary-{i}.json").read_text()) for i in range(len(variants))]
            mean = lambda k: float(np.mean([p[k] for p in parts]))  # noqa: E731
            fin = [p["finished"] for p in parts]
            comb = dict(parts[0], n=sum(p["n"] for p in parts), finished=None if None in fin else sum(fin),
                        mean_result=mean("mean_result"), worst_result=mean("worst_result"),
                        **({"result": mean("result")} if all(p.get("result") is not None for p in parts) else {}),
                        median_score=mean("median_score"), mean_score=mean("mean_score"), cpu_s=sum(p["cpu_s"] for p in parts),
                        rank_key=[float(np.mean([p["rank_key"][j] for p in parts])) for j in range(len(parts[0]["rank_key"]))],
                        variants={v: dict(mean=p["mean_result"], worst=p["worst_result"]) for v, p in zip(variants, parts)})
            f.write_text(json.dumps(comb, indent=1))
        if rc != 0 or not f.exists():
            err = f.parent / "stderr.txt"
            tail = err.read_text(errors="replace").strip().splitlines()[-5:] if err.exists() else []
            print(f"[duck-rl] TS eval of {name} failed (exit {rc})" + "".join(f"\n    {x}" for x in tail),
                  flush=True)
            return
        s = json.loads(f.read_text())
        result, result_n, rank_key = summary_rank(s, variants)
        rec = dict(onnx=cur["onnx"], steps=cur["steps"], stage=cur["stage"], kind=s["result_kind"], unit=s["result_unit"],
                   result=result, result_n=result_n,
                   mean=s["mean_result"], worst=s["worst_result"], finished=s["finished"], n=s["n"],
                   median_score=s["median_score"], mean_score=s["mean_score"],
                   rank_key=rank_key, out_before_limit=s["diagnostics"].get("out_before_limit"),
                   eval_s=round(time.time() - cur["t0"], 1), cpu_s=s["cpu_s"], t=time.strftime("%H:%M:%S"),
                   variants=s.get("variants"), world=s.get("world"), run_id=s.get("run_id"), run_dir=s.get("run_dir"),
                   # kit 1.6.2: the checkpoint's training seconds, and [seed, natural result] per seed (selection noise
                   # and best-of-k can be re-analysed offline; null with variants)
                   train_elapsed_s=round(float(cur["elapsed_s"]), 1) if cur.get("elapsed_s") is not None else None,
                   per_seed=None if variants else [[r.get("seed"), r.get("result")] + ([r["opponent"]] if "opponent" in r else [])
                                                   for r in s.get("per_seed") or []])
        if s.get("duel"):  # a duel Challenge: the rounds' tally, the BT estimate, per opponent (ts/duel_eval.ts)
            du = s["duel"]
            rec["duel"] = dict({k: du.get(k) for k in ("rounds", "wins", "draws", "losses", "invalid", "points")},
                               bt=(du.get("bt_estimate") or {}).get("rating"), run_rounds=(du.get("run") or {}).get("rounds"),
                               by_opponent={o: {k: v.get(k) for k in ("mean", "worst", "wins", "draws", "losses")}
                                            for o, v in (du.get("by_opponent") or {}).items()})
        if s.get("crowd"):  # a crowd Challenge (kit 1.11.0, EVAL_CHALLENGE): the rounds' wins and mean place (ts/crowd_eval.ts)
            cr, tie = s["crowd"], s["crowd"].get("tie_time") or {}
            rec["crowd"] = dict(vs=cr.get("vs"), ducks=cr.get("ducks"), wins=cr.get("wins"), mean_place=cr.get("mean_place"),
                                tie_kind=tie.get("kind"), tie_mean=tie.get("mean"))
        if self.busy is not None:  # N25: an eval that got far less CPU than the first one
            line = self.busy.eval(self.progress[-1][1] if self.progress else 0.0, rec["eval_s"], rec["cpu_s"])
            if line:
                print(f"[duck-rl] {line}", flush=True)
        if cur.get("start"):  # N26: the start's eval: kept apart (start_eval.json), never a checkpoint
            rec.update(onnx=None, start=cur["start"], steps=0, stage=None, train_elapsed_s=None)
            self.start_rec = rec
            with contextlib.suppress(OSError):
                (self.out / "start_eval.json").write_text(json.dumps(rec, indent=1))
            print(f"[duck-rl] TS eval of the start ({cur['start']}): {result_text(rec)} ({rec['eval_s']} s), the same seeds"
                  " and opponents as every checkpoint's", flush=True)
            return
        if cur.get("lr_cycle") is not None:  # LR_ANNEAL "cos3": the LR cycle that produced this checkpoint
            rec["lr_cycle"] = cur["lr_cycle"]
        key = self.cfg.get("select_key")
        if key:  # select by a per-seed Challenge diagnostic (ts/challenges/<id>.ts) instead of the Challenge's natural result: its median
            parts_ = [json.loads((f.parent / f"summary-{i}.json").read_text()) for i in range(len(variants))] if variants else [s]
            vals = [[float((r.get("challenge") or {}).get(key) or 0.0) for r in p_["per_seed"]] for p_ in parts_]
            med_ = float(np.mean([np.median(v) for v in vals]))
            rec.update(select_key=key, key_median=med_, rank_key=[med_, float(np.mean([min(v) for v in vals]))],
                       key_variants={v: float(np.median(x)) for v, x in zip(variants, vals)} if variants else None)
        if self.run_meta is not None and rec.get("run_dir"):
            self.settle(rec, self.best())
        self.records.append(rec)
        self.log.write(json.dumps(rec) + "\n")
        self.log.flush()
        b = self.best()
        ks = f", {key} median {rec['key_median']:.3f}" if key else ""
        run = (f" run {rec['run_id']}" if rec.get("run_kept") else " (run not kept: no improvement)"
               if rec.get("run_kept") is False else "")
        print(f"[duck-rl] TS eval {Path(cur['onnx']).name}: {result_text(rec)}{ks} ({rec['eval_s']} s)"
              f"{' <- best' if b is rec else ''}{run}", flush=True)
        if "run_mb" in rec and len(self.run_bytes) == 1:
            self.print_estimate(rec)

    def settle(self, rec: dict, prev: dict | None) -> None:
        """A finished eval's staged run: kept (moved into the run store) or held in staging (save_runs "improved", not
        better than `prev`, the best so far); the run held before it is deleted (a newer eval replaces it)."""
        staged = Path(rec["run_dir"])
        size = dir_bytes(staged)
        rec["run_mb"] = round(size / 1e6, 3)
        self.run_bytes.append(size)
        if self.held is not None:
            shutil.rmtree(self.held[1], ignore_errors=True)
            self.held = None
        rec["improved"] = prev is None or tuple(rec["rank_key"]) > tuple(prev["rank_key"])
        if self.save_mode == "all" or rec["improved"]:
            self.keep(rec, staged, size)
        else:
            rec["run_dir"], rec["run_kept"] = None, False
            self.held = (rec, staged, size)

    def keep(self, rec: dict, staged: Path, size: int) -> None:
        """Move a complete staged run dir into the run store (one rename on the same disk: `duckbench runs` sees all of
        it or nothing)."""
        dst = self.runs_dir / staged.name
        try:
            os.rename(staged, dst)
        except OSError as e:
            print(f"[duck-rl] could not move run {staged.name} into {self.runs_dir}: {e}", flush=True)
            rec["run_dir"], rec["run_kept"] = None, False
            return
        rec["run_dir"], rec["run_kept"] = str(dst), True
        self.kept += 1
        self.kept_bytes += size

    def print_estimate(self, rec: dict) -> None:
        """After the first saved run: its size and what saving every checkpoint would use at this eval cadence (one eval
        at a time, at most one per every_s, of a checkpoint that exists: every --ckpt-every steps)."""
        size = self.run_bytes[0]
        steps, el = self.progress[-1] if self.progress else (0, 0.0)
        ckpt_s = self.ckpt_every / (steps / el) if self.ckpt_every and steps > 0 and el > 0 else 0.0
        cadence = max(float(self.cfg["every_s"]), float(rec["eval_s"]), ckpt_s, 1.0)
        per_min = size * 60.0 / cadence
        run_total = (f", ~{mb(per_min * self.train_minutes)} MB over this {self.train_minutes:.0f}-min run"
                     if self.train_minutes and self.train_minutes < 1e5 else "")
        what = (f"{rec['duel']['run_rounds']} of its {rec['n']} rounds" if (rec.get("duel") or {}).get("run_rounds")
                else f"{rec['n']} seeds")
        head = f"[duck-rl] one checkpoint run = {mb(size)} MB ({what}); "
        rate = f"~{mb(per_min)} MB/min of training (~{mb(per_min * 60)} MB/hour) at one eval every {cadence:.0f} s{run_total}"
        if self.save_mode == "all":
            print(head + f"--save-runs all uses {rate}; --save-runs improved keeps only the improving ones", flush=True)
        else:
            print(head + f"--save-runs all would use {rate}; improved-only usually keeps a handful", flush=True)

    def runs_summary(self) -> dict:
        """stats.json checkpoint_runs: the mode, runs kept of the evals that saved one, their MB, what "all" would use."""
        return dict(mode=self.save_mode if self.run_meta is not None else "off", kept=self.kept, evaluated=len(self.run_bytes),
                    kept_mb=round(self.kept_bytes / 1e6, 2), all_mb_estimate=round(sum(self.run_bytes) / 1e6, 2),
                    dir=str(self.runs_dir) if self.runs_dir is not None else None)

    def tick(self, steps: int, elapsed_s: float | None = None, lr_cycle: int | None = None) -> None:
        if elapsed_s is not None:
            self.progress.append((int(steps), float(elapsed_s)))
        if lr_cycle is not None:
            self.cycles.append((int(steps), int(lr_cycle)))
        self.poll()
        ck = checkpoints(self.out)
        if ck:
            self.start(ck[-1], ckpt_steps(ck[-1]))

    def wait(self, max_s: float) -> None:
        t_end = time.time() + max(0.0, max_s)
        while self.proc is not None and time.time() < t_end:
            time.sleep(0.2)
            self.poll()

    def best(self) -> dict | None:
        return max(self.records, key=lambda r: tuple(r["rank_key"]), default=None)

    def mark_final(self, onnx: Path, ended: str | None = None) -> str | None:
        """train.final = true in the saved run of the exported weights' eval (if it saved one), and train.ended = `ended`
        when given ("error": a crash exported the checkpoints so far, kit 1.26.0); its run id, or None. The run may have
        gone up to the site already: its `.uploaded` marker goes, so the CLI's next sync sends the new run.json."""
        rec = next((r for r in reversed(self.records) if r["onnx"] == str(onnx)), None)
        if rec is not None and self.held is not None and self.held[0] is rec:  # select="last": keep its held run
            self.keep(rec, self.held[1], self.held[2])
            self.held = None
        if rec is None or not rec.get("run_dir"):
            return None
        try:
            f = Path(rec["run_dir"]) / "run.json"
            run = json.loads(f.read_text())
            run.setdefault("train", {})["final"] = True
            if ended is not None:
                run["train"]["ended"] = ended
            tmp = f.with_name("run.json.tmp")
            tmp.write_text(json.dumps(run, indent=1) + "\n")
            os.replace(tmp, f)
            with contextlib.suppress(FileNotFoundError):  # apps/cli/src/runSync.ts UPLOADED_FILE: pending again
                (f.parent / ".uploaded").unlink()
            return rec["run_id"]
        except (OSError, ValueError) as e:
            print(f"[duck-rl] could not mark run {rec.get('run_id')} final: {e}", flush=True)
            return None

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=5)
        self.log.close()
        if self.staging is not None:  # the held run, an unfinished eval's
            shutil.rmtree(self.staging, ignore_errors=True)
            with contextlib.suppress(OSError):
                self.staging.parent.rmdir()  # .staging, if no other session stages there


NONE_KEY = -1e9  # a null result's rank key: the worst (ts/eval.ts NONE)


def summary_rank(s: dict, variants: list | None = None) -> tuple[float | None, int | None, list]:
    """A duck-rl-eval/1 summary's ranking number, how many seeds it averages, and its rank key (kit 1.14.0, Emil
    2026-10-30). Solo: the best-half mean, `result` from ts/eval.ts (rank key [result] in the direction); a summary
    without it (an eval.ts before 1.14.0) is recomputed from its per-seed results (duck_rl/rank.py). Duel and crowd
    Challenges: their own mean points and rank key, unchanged."""
    if s.get("duel") or s.get("crowd"):
        return s["mean_result"], None, s["rank_key"]
    n = int(s["n"])
    k = (n + 1) // 2 if not variants else None
    lower = s.get("better") == "lower" or (s.get("better") is None and s.get("result_kind") == "race")
    if "result" in s:
        res = s["result"]
    elif not variants and s.get("per_seed"):
        res = rank_result([r.get("result") for r in s["per_seed"]], lower_is_better=lower)
    else:  # variants from an old eval.ts: no per-seed rows kept here
        res = s["mean_result"]
    return res, k, [NONE_KEY if res is None else (-res if lower else res)]


def result_text(r: dict) -> str:
    """A TS eval record's natural result: 'best 8 of 16 4.10 s (mean 4.21 s, finished 7/8, worst 6.90 s)' or
    'best 8 of 16 12.80 rpm (mean 12.30, worst 9.80 rpm)' (kit 1.14.0: the ranking number first)."""
    f = lambda v, p: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
    head = f"best {r['result_n']} of {r['n']}" if r.get("result_n") else "best half"
    if r["kind"] == "race":
        fin = f", finished {r['finished']}/{r['n']}" if r.get("finished") is not None else ""
        return f"{head} {f(r.get('result'), 2)} s (mean {f(r['mean'], 2)} s{fin}, worst {f(r['worst'], 2)} s)"
    if r.get("duel"):  # 'mean 0.719 pts per round (43 W 6 D 15 L of 64, BT ~1236), worst seed 0.250 [vs house:easy 0.72 ...]'
        du = r["duel"]
        bt = f", BT ~{du['bt']:.0f}" if du.get("bt") is not None else ""
        per = du.get("by_opponent") or {}
        vs = " [" + ", ".join(f"vs {o} {f(v['mean'], 3)}" for o, v in per.items()) + "]" if len(per) > 1 else ""
        return (f"mean {f(r['mean'], 3)} {r['unit']} per round ({du['wins']} W {du['draws']} D {du['losses']} L of "
                f"{du['rounds']}{bt}), worst seed {f(r['worst'], 3)}{vs}")
    if r.get("crowd"):  # 'mean 0.583 pts per round (1st in 6 of 16, mean place 2.10 of 4), worst seed 0.000'
        cr = r["crowd"]
        return (f"mean {f(r['mean'], 3)} {r['unit']} per round (1st in {cr['wins']} of {r['n']}, mean place "
                f"{f(cr['mean_place'], 2)} of {cr['ducks']}), worst seed {f(r['worst'], 3)}")
    return f"{head} {f(r.get('result'), 3)} {r['unit']} (mean {f(r['mean'], 3)}, worst {f(r['worst'], 3)} {r['unit']})"


def score_of(r: dict) -> float | None:
    """A TS eval record's ranking number: `result` (kit 1.14.0: the best-half mean; duel / crowd: the mean points), else
    (a record of an older kit) its mean."""
    return r["result"] if r.get("result") is not None else r.get("mean")


def learner_record(requested: str, summary: dict | None = None, hist: list[dict] = (), **rest) -> dict:
    """stats.json `learner` (PR #1 R5-5, R6-3): `name` the learner that actually ran ("mlx" or "cpu"), `requested` the one
    asked for (--learner, else the automatic pick: MLX where it is available), `fallback` why an MLX run fell back to the
    CPU learner mid-run (lab/overlap.py _fall_back on any MLX failure; L.train's summary["mlx_fallback"], or the log
    record that first carries it when the run ended without a summary) or None; then its knobs, --set overrides, recipe
    and resume. After a fallback the rest of the run is on the CPU learner, so `name` is "cpu"."""
    if requested not in ("mlx", "cpu"):
        raise ValueError(f"learner {requested!r}: mlx or cpu")
    fallback = (summary or {}).get("mlx_fallback") or next((h["mlx_fallback"] for h in hist if h.get("mlx_fallback")), None)
    return dict(name="cpu" if fallback else requested, requested=requested, fallback=fallback, **rest)


def run_card(hist: list[dict], records: list[dict], info: dict, targets, seed: int, recipe: str, knobs: dict,
             overrides: dict, train_s: float | None, steps: int, marks_min=(2, 5, 10)) -> dict:
    """The run's "time to target" card (kit 1.6.2; docs/research/training-speedup/README.md section 4), from the in-loop
    TS evals (the Challenge's ranking number over the eval seeds, the best-half mean since kit 1.14.0 (score_of); in the
    Challenge's direction) and challenge_log:
      to_frac_best   {50,80,90,95}: the first eval whose mean reached that share of the run's best mean (a race: a
                     time within best / share, i.e. that share of the best speed), with its env steps and training s
      to_frac_gain   the same for that share of the gain from the first eval to the best (the first eval is ~the
                     warm start, 250k steps in)
      to_target      the same for each absolute TARGETS value (a race: mean finish time <= the target)
      at_min         per 2/5/10 min of training: the last eval of a checkpoint trained by then, and the best so far
      best, last     the best eval by the rank key (what select="best" exports) and the last one; gap_last_vs_best
                     (> 0: the last is worse by that much)
      sps            median, first-minute and last-minute env steps/s and their ratio (heat, load); learner_share =
                     median t_update / median t_collect
    plus seed, recipe (train.py recipe_hash), the knobs that differ from the defaults and the --set overrides."""
    rk = info.get("ranking") or {}
    lower = bool(rk["lower_is_better"]) if "lower_is_better" in rk else bool(records and records[0].get("kind") == "race")
    sgn = -1.0 if lower else 1.0
    evs = sorted((r for r in records if score_of(r) is not None), key=lambda r: r["steps"])
    card = dict(direction="lower" if lower else "higher", unit=(records[0].get("unit") if records else None), seed=seed,
                recipe=recipe, knobs={n: v for n, v in knobs.items() if v != LEARNER_KNOBS[n]}, overrides=overrides,
                steps=steps, train_s=train_s, n_evals=len(evs))

    def pt(r: dict | None) -> dict | None:
        return None if r is None else dict(steps=int(r["steps"]), train_s=r.get("train_elapsed_s"), result=round(score_of(r), 4),
                                           mean=round(r["mean"], 4) if r.get("mean") is not None else None,
                                           worst=round(r["worst"], 4) if r.get("worst") is not None else None)

    def first(reached) -> dict | None:
        return pt(next((r for r in evs if reached(score_of(r))), None))
    if evs:
        bmean = (min if lower else max)(score_of(r) for r in evs)
        x0 = score_of(evs[0])
        fb, fg = {}, {}
        for f in (0.5, 0.8, 0.9, 0.95):
            key = str(int(round(f * 100)))
            if bmean > 0:
                thr = bmean / f if lower else f * bmean
                fb[key] = dict(threshold=round(thr, 4), at=first(lambda x, t=thr: sgn * (x - t) >= 0))
            gain = sgn * (bmean - x0)
            if gain > 0:
                fg[key] = dict(threshold=round(x0 + sgn * f * gain, 4), at=first(lambda x, ff=f: sgn * (x - x0) >= ff * gain))
        card.update(to_frac_best=fb or None, to_frac_gain=fg or None, first=pt(evs[0]),
                    to_target={str(t): first(lambda x, t=float(t): sgn * (x - t) >= 0) for t in targets} or None)
        at = {}
        for m in marks_min:
            if train_s is None or train_s < m * 60.0 - 1.0:
                continue
            by = [r for r in evs if r.get("train_elapsed_s") is not None and r["train_elapsed_s"] <= m * 60.0]
            at[str(m)] = dict(last=pt(by[-1]), best=pt(max(by, key=lambda r: tuple(r["rank_key"])))) if by else None
        best = max(evs, key=lambda r: tuple(r["rank_key"]))
        card.update(at_min=at or None, best=pt(best), last=pt(evs[-1]),
                    gap_last_vs_best=round(sgn * (score_of(best) - score_of(evs[-1])), 4))
    rows = [h for h in hist if h.get("sps")]
    if rows:
        end = rows[-1]
        early = [h for h in rows if h["elapsed_s"] <= 60.0] or rows[:1]
        late0 = next(h for h in rows if h["elapsed_s"] >= end["elapsed_s"] - 60.0)
        s_first = early[-1]["steps"] / early[-1]["elapsed_s"] if early[-1]["elapsed_s"] > 0 else None
        s_last = ((end["steps"] - late0["steps"]) / (end["elapsed_s"] - late0["elapsed_s"])
                  if end["elapsed_s"] > late0["elapsed_s"] else None)
        tu, tc = [h["t_update"] for h in rows if h.get("t_update")], [h["t_collect"] for h in rows if h.get("t_collect")]
        card["sps"] = dict(median=int(np.median([h["sps"] for h in rows])),
                           first_min=round(s_first) if s_first else None, last_min=round(s_last) if s_last else None,
                           last_vs_first=round(s_last / s_first, 3) if s_first and s_last else None,
                           learner_share=round(float(np.median(tu)) / float(np.median(tc)), 3) if tu and tc else None)
    return card


def card_line(c: dict) -> str:
    """The run card in one line for the log."""
    def at(p: dict | None) -> str:
        return "-" if not p else f"{p['steps'] / 1e6:.2f}M/{p['train_s'] or 0:.0f}s"
    u = c.get("unit") or ""
    parts = []
    if c.get("best"):
        parts.append(f"best {c['best']['mean']:.3f} {u} @ {at(c['best'])}, last {c['last']['mean']:.3f}")
    for k in ("50", "90"):
        x = (c.get("to_frac_gain") or {}).get(k)
        if x:
            parts.append(f"{k}% of the gain ({x['threshold']:.3f}) @ {at(x['at'])}")
    for t, p in (c.get("to_target") or {}).items():
        parts.append(f"{t} {u} @ {at(p)}")
    if c.get("sps"):
        parts.append(f"median {c['sps']['median']} sps (last/first min {c['sps']['last_vs_first']})")
    return "; ".join(parts) or "no in-loop evals"


def prior_curve(init: Path) -> dict | None:
    """The in-loop curve of the run a --resume checkpoint comes from (kit 1.8.1), for the whole continued run's
    "Curve:" line: {points: [[steps, train_s, mean], ...] up to the checkpoint, offset: {steps, train_s} = where this
    session starts in the run, sessions: the sessions so far}. From the `curve` the kit (>= 1.8.1) wrote beside the
    checkpoint (a submission's submission.json train.curve, a run's stats.json curve), else from that session's own
    in-loop evals when it continued nothing; None when unknown (the line then covers this session)."""
    init = init.resolve()
    meta = {}
    for f, key in ((init.parent / "submission.json", "train"), (init.parent / "stats.json", None)):
        if f.is_file():
            with contextlib.suppress(ValueError, OSError):
                d = json.loads(f.read_text())
                meta = d.get(key) if key else d
                if isinstance(meta, dict):
                    break
            meta = {}
    if not meta:
        return None
    try:
        ck = torch.load(init, map_location="cpu")
    except Exception:  # noqa: BLE001
        return None
    steps = ck.get("session_steps", ck.get("steps")) if isinstance(ck, dict) else None  # (1.15.0: names carry the run's)
    if steps is None:
        return None
    cur = meta.get("curve") if isinstance(meta.get("curve"), dict) else None
    if cur and isinstance(cur.get("points"), list) and isinstance(cur.get("offset"), dict):
        pts, off, sessions = cur["points"], cur["offset"], int(cur.get("sessions") or 1) - 1
    elif not meta.get("init") and isinstance(meta.get("ts_evals"), list):  # a first session (kit < 1.8.1): its own evals
        pts = [[r["steps"], r.get("train_elapsed_s"), score_of(r)] for r in meta["ts_evals"] if score_of(r) is not None]
        off, sessions = {"steps": 0, "train_s": 0.0}, 0
    else:
        return None
    s0, t0 = off.get("steps"), off.get("train_s")
    if s0 is None or t0 is None or any(p[1] is None for p in pts):
        return None
    end = s0 + int(steps)
    kept = [p for p in pts if p[0] <= end]
    el = ck.get("elapsed_s")
    at = [p[1] for p in kept if p[0] == end]  # the checkpoint's own score, if it was scored: its training s
    t_end = at[-1] if at else t0 + float(el) if el is not None else max((p[1] for p in kept), default=None)
    if t_end is None:
        return None
    return dict(points=kept, offset={"steps": end, "train_s": round(t_end, 1)}, sessions=sessions + 1)


def mmss(s: float) -> str:
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def short_text(r: dict) -> str:
    """A record's numbers in a few words: '0.531 pts per round (17 W 0 D 15 L of 32)', '4.10 s (finished 7/8)',
    '12.800 rpm'."""
    f = lambda v, p: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
    if r.get("duel"):
        du = r["duel"]
        return f"{f(score_of(r), 3)} {r['unit']} per round ({du['wins']} W {du['draws']} D {du['losses']} L of {du['rounds']})"
    if r.get("crowd"):
        return f"{f(score_of(r), 3)} {r['unit']} per round (1st in {r['crowd']['wins']} of {r['n']})"
    if r.get("kind") == "race":
        fin = f" (finished {r['finished']}/{r['n']})" if r.get("finished") is not None else ""
        return f"{f(score_of(r), 2)} s{fin}"
    return f"{f(score_of(r), 3)} {r['unit']}"


def start_vs_trained(start: dict | None, trained: dict | None, info: dict, seeds, opponents: list[str] | None,
                     start_note: str | None = None) -> dict:
    """N26 (newbie test 2026-11-02): the end's start -> trained comparison on the same seeds and opponents (the start's
    eval, TsEval.start_eval, and the exported weights' in-loop eval), so the agent needs no re-run. summary.json and
    stats.json `start_vs_trained`: {seeds, opponents, unit, direction, start, trained (each {what, result, text, n,
    rank_key, eval_s, + duel / finished}), verdict "better" / "same" / "worse" (null when a side is missing), change
    (trained - start in the Challenge's direction: > 0 is better), line (printed)}. "same": both show the same number at
    the log's precision (and, a duel, the same W / D / L)."""
    rk = info.get("ranking") or {}
    ref = trained or start or {}
    lower = bool(rk["lower_is_better"]) if "lower_is_better" in rk else ref.get("kind") == "race"
    dp = 2 if ref.get("kind") == "race" and not (ref.get("duel") or ref.get("crowd")) else 3

    def side(r: dict | None, what: str | None) -> dict | None:
        if r is None:
            return None
        d = dict(what=what, result=score_of(r), text=short_text(r), n=r.get("n"), rank_key=r.get("rank_key"),
                 eval_s=r.get("eval_s"))
        if r.get("duel"):
            d["duel"] = {k: r["duel"].get(k) for k in ("rounds", "wins", "draws", "losses")}
        if r.get("finished") is not None:
            d["finished"] = r["finished"]
        return d

    t_what = Path(trained["onnx"]).stem if trained and trained.get("onnx") else None
    out = dict(seeds=str(seeds), opponents=list(opponents or []), unit=ref.get("unit"),
               direction="lower" if lower else "higher", start=side(start, start.get("start") if start else None),
               trained=side(trained, t_what), verdict=None, change=None)
    on = f"same seeds {seeds}" + (f" vs {', '.join(opponents)}" if opponents else "")
    if start is None or trained is None or score_of(start) is None or score_of(trained) is None:
        why = start_note or ("the start was not scored" if start is None or score_of(start) is None
                             else "the exported weights were not scored")
        out["line"] = f"[duck-rl] start -> trained: no comparison ({why})."
        return out
    a, b = score_of(start), score_of(trained)
    same = f"{a:.{dp}f}" == f"{b:.{dp}f}" and (out["start"].get("duel") == out["trained"].get("duel"))
    change = (a - b) if lower else (b - a)
    out["change"] = round(change, 4)
    out["verdict"] = "same" if same else "better" if change > 0 else "worse"
    if out["verdict"] == "same":
        out["line"] = f"[duck-rl] Training didn't help: {short_text(trained)}, as before ({on}). No need to re-simulate."
    elif out["verdict"] == "worse":
        out["line"] = (f"[duck-rl] Training didn't help: start {short_text(start)} -> trained {short_text(trained)} ({on});"
                       " the start was better. No need to re-simulate.")
    else:
        out["line"] = (f"[duck-rl] Start -> trained ({on}): {short_text(start)} -> {short_text(trained)}."
                       " No need to re-simulate.")
    return out


def training_curve(records: list[dict], info: dict, train_s: float | None, prior: dict | None,
                   resumed: bool, start: dict | None = None) -> dict:
    """The run's in-loop curve in one neutral line (kit 1.8.1): the first checkpoint's score, the best score and when
    it was reached (training time and share of the run), and how much the best so far moved over the last third of
    the run. Scores are the in-loop evals' ranking numbers (ts_eval.jsonl; score_of: the best-half mean since kit 1.14.0,
    the mean before; a --resume of an older run continues its means) in the Challenge's unit and direction. With --resume and
    the earlier sessions' curve (prior_curve) it covers the whole continued run; with --resume and no such data,
    this session. Returns {line, scope, points (cumulative [steps, train_s, score]), offset, sessions, ...}."""
    rk = info.get("ranking") or {}
    evs = sorted((r for r in records if score_of(r) is not None and r.get("train_elapsed_s") is not None),
                 key=lambda r: r["steps"])
    lower = bool(rk["lower_is_better"]) if "lower_is_better" in rk else bool(evs and evs[0].get("kind") == "race")
    race = bool(evs and evs[0].get("kind") == "race")
    unit = (evs[0].get("unit") if evs else None) or ("s" if race else "")
    off = (prior or {}).get("offset") or {"steps": 0, "train_s": 0.0}
    pts = [list(p) for p in (prior or {}).get("points", [])]
    pts += [[int(off["steps"] + r["steps"]), round(off["train_s"] + r["train_elapsed_s"], 1), round(score_of(r), 4)]
            for r in evs]
    sessions = (prior or {}).get("sessions", 0) + 1
    scope = "run" if prior is not None or not resumed else "session"
    total = off["train_s"] + (train_s if train_s is not None else (evs[-1]["train_elapsed_s"] if evs else 0.0))
    head = ("Curve" + (f" (the whole run, {sessions} sessions)" if prior is not None and sessions > 1 else "")
            + (" (this session)" if scope == "session" else ""))
    out = dict(scope=scope, sessions=sessions, direction="lower" if lower else "higher", unit=unit,
               points=pts, offset=off, train_s=round(total, 1))
    dp = 3 if evs and (evs[0].get("duel") or evs[0].get("crowd")) else 2
    v = lambda x: f"{x:.{dp}f} {unit}".rstrip()  # noqa: E731
    if not pts:
        out["line"] = f"{head}: no checkpoint was scored in the loop."
        return out
    better = (lambda a, b: a < b) if lower else (lambda a, b: a > b)
    first, best = pts[0], pts[0]
    for p in pts[1:]:
        if better(p[2], best[2]):
            best = p
    frac = best[1] / total if total > 0 else None
    out.update(first=first[2], best=best[2], best_steps=best[0], best_train_s=best[1],
               best_frac=round(frac, 4) if frac is not None else None)
    # N17 (newbie test 2026-11-02): the start's score first (this session's start, TsEval.start_eval), and plainly
    # whether the best checkpoint beat it
    s0 = score_of(start) if start is not None and prior is None else None
    lead = f"start {v(s0)} (before training) → " if s0 is not None else ""
    beat = None if s0 is None else (f"{best[2]:.{dp}f}" != f"{s0:.{dp}f}" and better(best[2], s0))
    if s0 is not None:
        out.update(start=round(s0, 4), beat_start=beat)
    verdict = "" if beat is None else " The best checkpoint beats the start." if beat else " Training did not beat the start."
    line = (f"{head}: {lead}first checkpoint {v(first[2])} → best {v(best[2])} at {mmss(best[1])}"
            + (f" ({frac * 100:.0f} % of the {scope})" if frac is not None else ""))
    if len(pts) == 1:
        out["line"] = f"{head}: {lead}one checkpoint scored, {v(first[2])} at {mmss(first[1])}.{verdict}"
        return out
    cut = total * 2.0 / 3.0
    before = [p for p in pts if p[1] <= cut] or pts[:1]
    b0 = before[0][2]
    for p in before[1:]:
        if better(p[2], b0):
            b0 = p[2]
    gain = abs(best[2] - b0)
    pct = gain / abs(b0) * 100.0 if b0 else None
    out["last_third"] = {"from": b0, "to": best[2], "change": round(gain, 4),
                        "pct": round(pct, 2) if pct is not None else None}
    if gain > 0:
        line += (f"; last third of the {scope}: the best so far improved by {v(gain)}"
                 + (f" ({pct:.1f} %)" if pct is not None else "") + ".")
    else:
        line += f"; last third of the {scope}: the best so far did not change (the training had stopped improving)."
    out["line"] = line + verdict
    return out


def main() -> int:
    ap = build_parser()
    a = ap.parse_args()
    if a.detach and DETACHED is None:  # an abbreviation (--det): the early check takes --detach only in full
        ap.error("--detach: write it in full")
    if a.learner is None:  # kit 1.15.0: no "MLX learner failed" warning on Linux, where the CPU learner is the plan
        from lab.overlap import mlx_available
        a.learner = "mlx" if mlx_available() else "cpu"
        if a.learner == "cpu":
            print("[duck-rl] learner: cpu (MLX runs on Apple silicon only)", flush=True)
    if a.init and a.resume:
        ap.error("--init or --resume, not both")
    a.model = (a.model or os.environ.get("DUCKBENCH_AGENT_MODEL") or "").strip() or saved_agent_model()
    if a.model is not None and not MODEL_RE.fullmatch(a.model):  # the server would drop it too (trainModelName)
        print(f"[duck-rl] warning: model {a.model!r} (--model / DUCKBENCH_AGENT_MODEL) is not 1-64 chars of "
              "[A-Za-z0-9 ._@/+:-]: train.model is left out (pass --model to name it)", file=sys.stderr, flush=True)
        a.model = None
    if a.model is None:
        print(f"[duck-rl] train.model: none yet (no --model, no DUCKBENCH_AGENT_MODEL, no model in {player_home() / 'agent.json'});"
              " the export takes it from there if the CLI saves one meanwhile; else pass --model <your model>", flush=True)
    if a.seed is None:
        a.seed = secrets.randbelow(2**31) if bridge.STANDALONE else 1
    layout = "given"
    if a.workers is None:  # kit 1.6.0 (LB-56): the efficiency cores and the priority go to the person's apps
        a.workers, _, layout = device.default_layout()
    if a.per_worker is None:
        a.per_worker = max(1, round(device.BATCH_ENVS / a.workers))
    if a.nice > 0:
        os.nice(a.nice)  # inherited by the env workers, the learner and the TS eval
    a.layout = f"{a.workers}x{a.per_worker} envs ({layout}), nice {a.nice}"
    load_note(a.workers)
    if a.learner == "cpu":
        cpu_learner_threads(a)
    with budget.charged("train", a.label, start=T0) as brun:  # the owner's Mac-time budget in a player's kit
        return run(a, brun)


def cpu_learner_threads(a: argparse.Namespace) -> None:
    """kit 1.15.0: torch's default (every core) competed with the env workers (~10 cores on 32)."""
    import os as _os
    import torch
    free = max(1, (_os.cpu_count() or 1) - a.workers)
    n = a.learner_threads or max(1, min(4, free // 2))
    torch.set_num_threads(n)
    print(f"[duck-rl] CPU learner: {n} torch thread(s) (--learner-threads; {a.workers} env workers)", flush=True)


def post_end(session: str, tse, error: str | None) -> None:
    """The training's end on the human's page (`duckbench kit done`, bridge.post_done): "Training done" when it ended
    well and kept a checkpoint run; "Training crashed: <error>" on a crash, with or without one (kit 1.26.0). Never fails."""
    if error is None and (tse is None or tse.run_meta is None or tse.kept == 0):
        return
    runs = tse.runs_dir if tse is not None and tse.runs_dir is not None else runs_dir()
    note = bridge.post_done(session, runs, crashed=status.one_line(error) if error is not None else None)
    if note:
        print(f"[duck-rl] site: {note}", flush=True)


def run(a: argparse.Namespace, brun: budget.Run | None) -> int:
    global STEP_OFFSET
    signal.signal(signal.SIGTERM, _on_term)
    t_start = T0

    cfg_path = Path(a.config).resolve()
    cfg_mod = load_config(cfg_path)
    overrides = apply_overrides(cfg_mod, a.sets)
    if callable(getattr(cfg_mod, "after_set", None)):  # kit 1.17.0: a config that derives knobs from others (configs/base/move.py)
        cfg_mod.after_set()
    # (PR #1 B5: the MLX learner's batch follows the nets' widths, lab/batch_layout.py, so a wider obs, as ENCODER's 61 + its numbers, or
    # ENCODER's 61 + its numbers, learns on it too; kit 1.25.0-1.27.0 moved such a training to the CPU learner)
    global SITE_ENCODER
    SITE_ENCODER = config_encoder(cfg_mod)
    if SITE_ENCODER is not None:
        e = SITE_ENCODER
        print(f"[duck-rl] ENCODER {e.name} ({e.id}, frozen): the obs is the 61 + its {e.out} named numbers of the camera "
              f"picture ({e.macs:,} MACs per refresh, every program tick); the export is the site's camera move", flush=True)
    knobs = learner_knobs(cfg_mod, a.learner)
    kl_sched = kl_schedule(cfg_mod, knobs)  # experimental DESIRED_KL_END (None: no schedule, the run as before)
    resume = bool(a.resume)
    if resume:
        a.init = a.resume  # the same lineage and checkpoint handling as --init; L.train(resume=True) adds the rest
    if a.init and (why := parent_encoder_problem(Path(a.init), "resume" if resume else "init", SITE_ENCODER)):
        raise SystemExit(why)  # PR #1 R12-1: before the run dir moves aside or anything trains
    challenge = config_challenge(cfg_mod, cfg_path)
    info = duck_rl.load_info(challenge)
    if SITE_ENCODER is not None and "duel" in info:
        raise SystemExit(f"ENCODER: {challenge} is a duel Challenge; the encoder trains on solo Challenges (a duel learner's obs is the 61)")
    # kit 1.11.0: EVAL_CHALLENGE scores the checkpoints and stamps the export (default: the training Challenge itself)
    eval_challenge = config_eval_challenge(cfg_mod, challenge)
    eval_info = info if eval_challenge == challenge else duck_rl.load_info(eval_challenge)
    if eval_challenge != challenge:
        if eval_info["challenge"]["version"] < 1:
            raise SystemExit(f"EVAL_CHALLENGE {eval_challenge!r}: a spike challenge (version 0) has no duck-strategy/1 export")
        if "duel" not in eval_info and getattr(cfg_mod, "EVAL_OPPONENTS", None) is not None:
            raise SystemExit(f"EVAL_OPPONENTS: EVAL_CHALLENGE {eval_challenge!r} is not a duel challenge (a crowd challenge plays its own field)")
    out = duck_rl.RUNS / a.label
    prior = prior_curve(Path(a.resume)) if resume else None  # kit 1.8.1: the earlier sessions' curve, for the "Curve:" line
    sessions = (prior["sessions"] if prior else 0) + 1  # this session's number in the run
    pbest = prior_best(Path(a.resume), sessions - 1) if resume else None  # (read before the run dir moves aside)
    mine = DETACHED is not None and Path(DETACHED) == out  # --detach made it (it holds this run's train.log)
    if out.exists() and not mine:
        refuse_live(out, a.label)
        shutil.move(str(out), str(out.with_name(f"{a.label}.old-{time.strftime('%H%M%S')}")))
    out.mkdir(parents=True, exist_ok=mine)
    # kit 1.26.0: status.json beside train.log: running (with a heartbeat), then done / stopped / crashed (__main__)
    STATUS["s"] = status.Status(out, a.label, session=f"{a.label}-{time.strftime('%Y%m%d-%H%M%S', time.localtime(T0))}",
                                started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(T0))).start()
    fixed_moves = node_moves(cfg_mod, challenge, out)  # kit 1.19.0: a PROGRAM with moves on its nodes ([] for a plain one)
    program = resolve(cfg_mod.PROGRAM)
    compiled = compile_program(challenge, program, out / "compiled.json")
    nodes = trained_nodes(cfg_mod)
    # kit 1.24.0: a node gets one move: PROGRAM's own (node_moves) or BASE_STRATEGY's, never both
    both = sorted({n for m in fixed_moves for n in m["nodes"]} & {n for e in base_moves(cfg_mod, nodes) for n in e["nodes"]})
    if both:
        raise SystemExit(f"node(s) {', '.join(both)}: PROGRAM {cfg_mod.PROGRAM} binds a move there and BASE_STRATEGY "
                         f"{cfg_mod.BASE_STRATEGY} too: drop the move from one of them")
    env_cfg = env_config(cfg_mod, challenge, compiled)
    if knobs["STAGGER_START"]:
        env_cfg["stagger_start"] = True
    hc = host_config(cfg_mod, nodes, config_knobs(cfg_mod) if getattr(cfg_mod, "HOST", None) == "raw" else ())
    if hc["move_extra"] and eval_info["challenge"]["version"] < 1:
        raise SystemExit(f"spike challenge {challenge} (version 0): the raw host and DONE need a duck-strategy/1 export (a challenge version >= 1)")
    dev = device.resolve(a.device, getattr(cfg_mod, "DEVICE", None))  # train.device (auto-detection cached, ~0.5 s)
    session = STATUS["s"].rec["session"]  # this invocation, <label>-<YYYYMMDD-HHMMSS> of T0 (the saved runs' train.session)
    # kit 1.22.0 (LB-277): what this training runs, beside its logs: config.py, kit_files.json, kit_diff.patch (provenance.py)
    provenance.write(out, cfg_path, duck_rl.KIT, session=session, label=a.label, kit_version=duck_rl.KIT_VERSION, mode=bridge.MODE,
                     config=bridge.CONFIG, overrides=overrides,
                     encoder=dict(name=SITE_ENCODER.name, id=SITE_ENCODER.id) if SITE_ENCODER is not None else None)
    if hc["host"] == "raw":
        print(f"[duck-rl] raw host: command {hc['export']}" + (f", ranges {hc['ranges']} (per episode)" if hc["ranges"] else ""), flush=True)
    params = dict(getattr(cfg_mod, "REWARD_PARAMS", {}))
    names, episode_s = term_names(cfg_mod, params, env_cfg)
    warm = getattr(cfg_mod, "WARM_START", "alpha_walking")
    warm_mlp = getattr(cfg_mod, "WARM_MLP", None)
    warm_label, warm_mv = warm, None
    if warm_mlp:  # a path, a move id or an alias (moves.py); the label names one .mlp.json path (tools/moves.ts draftCard reads it)
        warm_mv = resolve_warm_mlp(str(warm_mlp))
        if SITE_ENCODER is not None and json.loads(Path(warm_mv["json"]).read_text()).get("input_dim") != 61:
            raise SystemExit(f"WARM_MLP {warm_mlp}: an ENCODER training warm-starts from a 61-input move (its weights go on "
                             "the 61, zero on the encoder's numbers); continue an encoder training with --resume or --init")
        L.warm_start = mlp_warm_start(Path(warm_mv["json"]))
        shown = f"{warm_mlp} = {warm_mv['json']}" if moves.is_library_ref(str(warm_mlp)) else str(warm_mlp)
        warm_label = f"{warm} graph, weights {shown} ({warm_mv['id']}{', ' + warm_mv['alias'] if warm_mv.get('alias') else ''})"
        print(f"[duck-rl] WARM_MLP {warm_mlp} -> {warm_mv['id']} ({warm_mv.get('alias') or 'no alias'}): {warm_mv['json']}", flush=True)
    init = init_parent(Path(a.init)) if a.init else None
    if init is not None:
        print(f"[duck-rl] --{'resume' if resume else 'init'} {a.init} -> weights {init['id']} at {init['run_steps']} steps", flush=True)
    if resume:
        STEP_OFFSET = int(prior["offset"]["steps"]) if prior else int(init["run_steps"] or 0)
        if pbest is not None:
            (out / "run-best").mkdir()
            for n, b in pbest["files"].items():
                (out / "run-best" / n).write_bytes(b)
            print(f"[duck-rl] whole-run best so far: {pbest['ckpt']} of session {pbest['session']} "
                  f"({result_text(pbest['rec'])}), copied to {out / 'run-best'}; the export picks it unless this "
                  "session scores better", flush=True)
        print(f"[duck-rl] steps: this session's checkpoints continue the run's count from {STEP_OFFSET:,} "
              f"(ckpt-<run steps>; session {sessions})", flush=True)
        print("[duck-rl] curve: " + (f"continues the run's curve ({len(prior['points'])} checkpoint scores, session "
                                     f"{prior['sessions'] + 1}, from {prior['offset']['steps']:,} steps)" if prior
                                     else "no earlier curve beside the checkpoint: the Curve line covers this session"),
              flush=True)
    parent = moves.parent_of(warm, warm_mv, init)
    print(f"[duck-rl] lineage: parent {parent['parent']}{' (' + parent['parent_alias'] + ')' if parent.get('parent_alias') else ''}, "
          f"root {parent['root']}, parent cumulative {parent['base']}" + (f"; {'; '.join(parent['notes'])}" if parent["notes"] else ""),
          flush=True)

    def deadline_now() -> float | None:
        """The earlier of the experiment runner's deadline and the owner's limit (a player's kit, duck_rl/budget.py:
        re-read; the kit's other charged commands running count)."""
        ds = [float(d) for d in (os.environ.get("DUCK_RL_DEADLINE_EPOCH"), brun.deadline() if brun else None) if d]
        return min(ds) if ds else None

    minutes = a.minutes
    deadline = deadline_now()
    if deadline:
        minutes = min(minutes, (deadline - time.time() - a.reserve_s) / 60.0)
    if minutes <= 0.05:
        raise SystemExit(f"not enough budget left to train ({minutes * 60:.0f} s after the export reserve)")
    z = np.zeros(13)
    move = ChallengeMove(env_cfg, path=cfg_path, slug=f"{challenge}-{a.label}", warm_start=warm, lo=z, hi=z, eval_cmd=z,
                    episode_s=episode_s, terminations={}, terms={n: "bonus" for n in names}, curriculum=[],
                    reward_terms=reward_fn(cfg_mod, params, names))
    # the rollout path (kit 1.8.0, duck_rl/rollout.py): --rollout, else the config's ROLLOUT, else fast
    ro = a.rollout or getattr(cfg_mod, "ROLLOUT", None) or "fast"
    ro_opts = rollout.options(ro)
    # (torch threads: the MLX learner's Updater runs the trainer's torch at 1 thread during training, as before, and
    # restores the count it found for a CPU fallback; the CPU learner keeps torch's own count)
    L.PoolVecEnv, L.FastPool, L.unpack = ChallengeVecEnv, functools.partial(ChallengePool, **ro_opts), make_unpack(info)
    ro_desc = rollout.describe(ro, a.per_worker)
    print(f"[duck-rl] rollout: {ro_desc}", flush=True)
    node_desc = (f"raw host, command {hc['export']}" if hc["host"] == "raw"
                 else compiled["nodes"][nodes[0]])  # a raw node's walk params never reach the move
    fed = (json.loads(program.read_text()).get("nodes", {}).get(nodes[0], {}).get("feed") or {})
    if fed:  # duck-program/2: the node's command comes from its feed every program tick
        node_desc = f"{node_desc}, fed {', '.join(fed)} by the duckscript"
    print(f"[duck-rl] kit {duck_rl.KIT_VERSION} {challenge} v{info['challenge']['version']} {a.label}: nodes {nodes} "
          f"({node_desc}), warm start {warm_label}, init {a.init or 'stock'}, terms {names}, episode {episode_s} s, "
          f"{minutes:.2f} min, {a.layout}, seed {a.seed}, device {dev or '-'} -> {out}", flush=True)
    changed = {n: v for n, v in knobs.items() if v != LEARNER_KNOBS[n]}
    run_shape = dict(workers=a.workers, per_worker=a.per_worker, learner=a.learner, ckpt_every=a.ckpt_every,
                     init=bool(a.init) and not resume, resume=resume, overrides=overrides)
    recipe = recipe_hash(knobs, run_shape, cfg_path)
    print(f"[duck-rl] learner: {', '.join(f'{n} {v}' for n, v in changed.items()) or 'the kit defaults'}"
          + (f" (--set {overrides})" if overrides else "") + f"; recipe {recipe}", flush=True)

    hist = []
    dlog = open(out / "dare_log.jsonl", "w")
    duel = env_cfg.get("duel")  # a duel Challenge (duel_config): its eval opponents, the per-iteration W / D / L
    ts_cfg = {**TS_EVAL_DEFAULTS, **(TS_EVAL_DUEL if duel else {}), **getattr(cfg_mod, "TS_EVAL", {})}
    for k in ("seeds", "every_s", "select"):
        v = getattr(a, f"ts_{k}")
        if v is not None:
            ts_cfg[k] = v
    if a.no_ts_eval:
        ts_cfg["enabled"] = False
    ts_cfg["save_runs"] = save_runs_mode(a.save_runs or ts_cfg.get("save_runs", "improved"))
    # the saved runs' train block (a real Challenge only: spike Challenges have no run the viewer can replay; a duel Challenge's run is a
    # match run, ts/duel_eval.ts)
    run_meta = (dict(session=session, label=a.label, kit_version=duck_rl.KIT_VERSION, **({"model": a.model} if a.model else {}),
                     **({"device": dev} if dev else {})) if eval_info["challenge"]["version"] >= 1 else None)
    # the eval's --opponent refs: a duel Challenge's own (EVAL_OPPONENTS / OPPONENT); none for a crowd or solo EVAL_CHALLENGE
    eval_opps = duel["eval"] if duel and "duel" in eval_info else None
    base = base_moves(cfg_mod, nodes)
    if base:
        print(f"[duck-rl] BASE_STRATEGY {cfg_mod.BASE_STRATEGY}: " + "; ".join(f"{','.join(e['nodes'])} <- {Path(e['move']).name}" for e in base)
              + " (they run while this move trains, and go into the export)", flush=True)
    tse = (TsEval(ts_cfg, out, eval_challenge, program, nodes, hc["move_extra"], run_meta, a.ckpt_every,
                  opponents=eval_opps, fixed_moves=fixed_moves, base=base) if ts_cfg["enabled"] else None)
    eval_vs = (f"vs {', '.join(eval_opps)}" if eval_opps else "")
    if eval_challenge != challenge:
        eval_vs = f"on {eval_challenge} v{eval_info['challenge']['version']}" + (f" {eval_vs}" if eval_vs else "")
        print(f"[duck-rl] EVAL_CHALLENGE {eval_challenge}: training on {challenge}, the checkpoints scored and the export stamped for "
              f"{eval_challenge}", flush=True)
    if duel:
        opp = ", ".join(f"{o['ref']} x{o['weight']:g}" for o in duel["opponents"])
        print(f"[duck-rl] duel: the learner (side A, either slot) vs {opp}; UNBOUND {duel['unbound']}; the in-loop TS "
              f"eval {eval_vs}", flush=True)
    if tse is not None:
        tse.train_end = time.time() + minutes * 60.0
        tse.train_minutes = minutes
        print(f"[duck-rl] in-loop TS eval: {ts_cfg}", flush=True)
        mode = tse.save_mode
        if tse.run_meta is None and mode != "off":
            print("[duck-rl] checkpoint runs: none (a spike challenge or variants: no run the viewer can replay)", flush=True)
        elif mode == "improved":
            print(f"[duck-rl] checkpoint runs: saving only checkpoints that improve (+ the final one) to {tse.runs_dir}; "
                  "--save-runs all saves every checkpoint for the viewer (more disk), --save-runs off none", flush=True)
        elif mode == "all":
            print(f"[duck-rl] checkpoint runs: saving every checkpoint to {tse.runs_dir} (more disk); --save-runs improved "
                  "keeps only the improving ones (+ the final one), --save-runs off none", flush=True)
        else:
            print("[duck-rl] checkpoint runs: off; --save-runs improved keeps the improving checkpoints for the viewer "
                  "(`duckbench view`)", flush=True)
        if eval_opps and tse.run_meta is not None:
            print(f"[duck-rl] a checkpoint run is a match: vs {eval_opps[0]} on the first 4 eval seeds from both sides "
                  "(the score is the whole eval's)", flush=True)
        elif eval_info["metric"]["kind"] == "field_points" and tse.run_meta is not None:
            print("[duck-rl] a checkpoint run is a crowd run: the first 4 eval seeds' rounds (the score is the whole "
                  "eval's)", flush=True)
    watch = busy.BusyWatch()  # N25: a sustained slowdown against this run's own start: one line, information only
    start_note = None  # N26: why there is no start eval, if there is none
    if tse is not None:
        tse.busy = watch
        if init is not None:  # the weights start from a checkpoint: an earlier session's numbers are its start
            start_note = (f"--{'resume' if resume else 'init'}: the run starts from {Path(a.init).name}; its own session's"
                          " summary has its numbers")
        elif tse.cfg.get("variants") or tse.cfg.get("select_key"):
            start_note = "variants / select_key evals score no plain result"
        else:  # the start = warm_start.json: WARM_MLP's weights on the trained nodes, else their stock skills
            what = (warm_mv.get("alias") or warm_mv["id"]) if warm_mv is not None else "stock skills"
            if tse.start_eval(Path(warm_mv["json"]) if warm_mv is not None else None, what):
                print(f"[duck-rl] scoring the start ({what}) on the eval seeds first, for the end's start -> trained "
                      "comparison", flush=True)

    def log_fn(rec: dict) -> str | None:
        if STOP["sig"] is not None:
            raise StopTraining()
        if "it" not in rec:
            print(f"[duck-rl] {rec}", flush=True)
            return
        env = ChallengeVecEnv.CURRENT
        eps = env.acc["eps"]
        rets, env.done_returns = env.done_returns, []
        mean = lambda k: float(np.nanmean([e[k] for e in eps])) if eps else None  # noqa: E731
        r = dict(it=rec["it"], steps=rec["steps"], elapsed_s=rec["elapsed_s"], sps=rec["sps"], ep_n=len(eps),
                 ep_len=rec["ep_len"], fall=mean("fell"), lost=mean("lost"), score=mean("score"), success=mean("success"),
                 ep_return=float(np.mean(rets)) if rets else None, reward_step=rec["reward_step"],
                 std=rec["std"], lr=rec["lr"], v_loss=rec["v_loss"], terms=rec["terms"], ckpt=rec.get("ckpt"))
        # kit 1.6.2: episode ends by reason, the workers' straggler stats, and log.jsonl's timing / update fields (the
        # update's of batch upd_it: it - 1 when overlapped)
        reasons = [e.get("reason") for e in eps]
        r.update({f"n_{k}": reasons.count(k) for k in (DUEL_END_REASONS if duel else END_REASONS)})
        if duel:  # the training side's W / D / L (duck_rl/duel_env.py end_reason), per opponent when there are several
            dec = [e["score"] for e in eps if e.get("reason") in ("won", "draw", "lost", "invalid")]
            r["points"] = float(np.mean(dec)) if dec else None  # mean points of the decided episodes
            if len(duel["opponents"]) > 1:
                r["by_opponent"] = {o["ref"]: {k: sum(1 for e in eps if e.get("opponent") == o["ref"] and e.get("reason") == k)
                                               for k in DUEL_END_REASONS} for o in duel["opponents"]}
        r.update(env.pool.timing() if hasattr(env.pool, "timing") else {})
        r.update({k: rec[k] for k in DLOG_COPY if k in rec})
        if kl_sched is not None:  # DESIRED_KL_END: the KL target of the next update (linear in the elapsed share)
            r["dkl"] = kl_sched.advance(rec["elapsed_s"] / (minutes * 60.0))
        hist.append(r)
        dlog.write(json.dumps(r) + "\n")
        dlog.flush()
        if tse is not None:
            tse.tick(r["steps"], r["elapsed_s"], r.get("lr_cycle"))
        if rec["it"] == 10:
            slow_machine_note(r["sps"], a.workers)
        busy_now = watch.sps(r["elapsed_s"], r["sps"], rec["it"])
        if busy_now:
            print(f"[duck-rl] {busy_now}", flush=True)
        if duel and (rec["it"] % 10 == 0 or rec.get("ckpt")):
            f = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
            print(f"it {r['it']:4d} steps {r['steps']:>9d} t {r['elapsed_s']:6.0f}s sps {r['sps']:6d} eps {r['ep_n']:3d} "
                  f"len {f(r['ep_len'], 0)} W/D/L {r['n_won']}/{r['n_draw']}/{r['n_lost']} unbound {r['n_unbound']}"
                  + (f" cut {r['n_timeout']}" if r["n_timeout"] else "") + (f" invalid {r['n_invalid']}" if r["n_invalid"] else "")
                  + f" pts {f(r['points'], 2)} ret {f(r['ep_return'], 2)} std {r['std']:.3f} lr {r['lr']:.1e}" + (" (warm-up: critic only)" if rec.get("warmup") else "")
                  + (f" cycle {r['lr_cycle']}" if r.get("lr_cycle") is not None else "")
                  + (f" ckpt {r['ckpt']}" if r["ckpt"] else ""), flush=True)
        elif rec["it"] % 10 == 0 or rec.get("ckpt"):
            f = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
            print(f"it {r['it']:4d} steps {r['steps']:>9d} t {r['elapsed_s']:6.0f}s sps {r['sps']:6d} eps {r['ep_n']:3d} "
                  f"len {f(r['ep_len'], 0)} fall {f(r['fall'], 2)} lost {f(r['lost'], 2)} metric@end {f(r['score'], 2)} "
                  f"success {f(r['success'], 2)} ret {f(r['ep_return'], 2)} std {r['std']:.3f} lr {r['lr']:.1e}" + (" (warm-up: critic only)" if rec.get("warmup") else "")
                  + (f" cycle {r['lr_cycle']}" if r.get("lr_cycle") is not None else "")
                  + (f" ckpt {r['ckpt']}" if r["ckpt"] else ""), flush=True)
        # the deadline is absolute (the setup before the loop counts) and re-read: the loop stops --reserve-s before it
        dl, end = deadline_now(), time.time() - rec["elapsed_s"] + minutes * 60.0
        if dl is not None:
            end = min(end, dl - a.reserve_s)
        if tse is not None:
            tse.train_end = end
        if dl is not None and time.time() >= dl - a.reserve_s:
            print(f"[duck-rl] {a.reserve_s:.0f} s to the budget deadline: training stops for the export", flush=True)
            return "deadline"
        return None

    init_critic = bool(a.init) and "critic" in torch.load(a.init, map_location="cpu")
    if kl_sched is not None:
        kl_sched.install()
        print(f"[duck-rl] DESIRED_KL_END (experimental): the KL target falls linearly from {kl_sched.start:g} to "
              f"{kl_sched.end:g} over the {minutes:.2f}-min budget (wall-clock share, steps of {KL_STEP:g}); dkl in "
              "dare_log.jsonl", flush=True)
    stopped, error = None, None
    summary = {}
    k = knobs
    try:
        summary = L.train(move, out, minutes=minutes, workers=a.workers, per_worker=a.per_worker, seed=a.seed,
                          init_from=a.init, init_critic=init_critic, final_curriculum=True, learner=a.learner,
                          overlap=a.learner == "mlx", ckpt_every_steps=a.ckpt_every, probe=None, log_fn=log_fn,
                          init_std=k["INIT_STD"], lr=k["LR"], lr_fixed=k["LR_FIXED"], warmup_iters=k["WARMUP_ITERS"],
                          desired_kl=k["DESIRED_KL"], critic_lr=k["CRITIC_LR"], critic_epochs=k["CRITIC_EPOCHS"],
                          value_clip=k["VALUE_CLIP"], rollout_steps=k["ROLLOUT_STEPS"], gamma=k["GAMMA"],
                          lam=k["LAMBDA"], std_floor=k["STD_FLOOR"], warmup_ev=k["WARMUP_EV"],
                          value_bias_init=k["VALUE_BIAS_INIT"], lr_anneal=k["LR_ANNEAL"], resume=resume,
                          step_offset=STEP_OFFSET)
        last = out / "policy.onnx"
    except Exception as e:  # noqa: BLE001  (StopTraining, the pool torn down around a SIGTERM, or a crash)
        if getattr(e, "mlx_fallback", None):  # lab/learn.py _carry_fallback: the reason a raise can't return (R7-2)
            summary = {"mlx_fallback": e.mlx_fallback}
        if STOP["sig"] is not None:
            if not isinstance(e, StopTraining):
                print(f"[duck-rl] SIGTERM during {type(e).__name__}: {e}", flush=True)
            stopped = "sigterm"
        else:  # kit 1.8.6: a crash still writes stats.json and the submission from the checkpoints saved so far (exit 1)
            import traceback
            traceback.print_exc()
            stopped, error = "error", f"{type(e).__name__}: {e}"
            STATUS["error"] = error
            print(f"[duck-rl] training crashed ({error}): stats.json and the submission come from the checkpoints "
                  "saved so far; exit 1", flush=True)
        ck = checkpoints(out)
        if not ck:
            if error is not None:
                dlog.close()
                (out / "stats.json").write_text(json.dumps(dict(
                    kit_version=duck_rl.KIT_VERSION, challenge=challenge, label=a.label, config=str(a.config), seed=a.seed,
                    nodes=nodes, host=hc["host"], stopped=stopped, ended="error", error=error, learner=learner_record(a.learner, summary, hist),
                    steps=hist[-1]["steps"] if hist else 0, ts_evals=[], last=hist[-1] if hist else None), indent=1))
                print("[duck-rl] crashed before the first checkpoint: nothing to submit", flush=True)
                post_end(session, tse, error)
                return 1
            print("[duck-rl] SIGTERM before the first checkpoint: nothing to submit", flush=True)
            return 3
        last = ck[-1]
    dlog.close()
    steps_done = summary.get("steps", hist[-1]["steps"] if hist else 0)
    onnx, choice, prior_won = last, "last", False
    if tse is not None:
        dl = deadline_now()
        if stopped is None and tse.cfg["final"]:  # the final weights too, if the deadline allows
            # An in-flight eval of an older checkpoint may still run: let it finish, then start the final one if the
            # time left (to 8 s before the deadline) covers the fastest eval so far (they overlapped training: niced).
            est = min((r["eval_s"] for r in tse.records), default=0.0)
            while deadline_left(dl) - 8.0 > est:
                tse.poll()
                if tse.proc is None:
                    tse.start(last, steps_done, force=True,
                              elapsed_s=summary.get("train_s") or (hist[-1]["elapsed_s"] if hist else None))
                    break
                time.sleep(0.2)
                est = min((r["eval_s"] for r in tse.records), default=0.0)
            if last not in tse.done:
                print(f"[duck-rl] final weights not TS-scored: {deadline_left(dl) - 8.0:.1f} s left before the export, "
                      f"the fastest eval took {est:.1f} s (--reserve-s gives the final eval more time)", flush=True)
        tse.wait(3.0 if stopped == "sigterm" else deadline_left(dl) - 8.0)  # (a crash lets an in-flight eval finish)
        if tse.proc is not None and tse.proc.poll() is None:
            print(f"[duck-rl] TS eval of {Path(tse.cur['onnx']).name} not finished before the export: dropped", flush=True)
        best = tse.best()
        if tse.cfg["select"] == "best" and best is not None:
            onnx, choice = Path(best["onnx"]), (f"best by TS ({result_text(best)}, "
                                                f"rank key {[round(x, 4) for x in best['rank_key']]})")
        if tse.cfg["select"] == "best" and pbest is not None:  # kit 1.15.0: the best of the whole run
            pr = pbest["rec"]
            if best is not None and not same_eval(pr, best):
                print(f"[duck-rl] whole-run best: session {pbest['session']}'s {pbest['ckpt']} was scored on other "
                      "seeds or opponents: the export picks from this session", flush=True)
            elif best is None or tuple(pr["rank_key"]) > tuple(best["rank_key"]):
                w = next(iter(pbest["files"]))  # (the weights first)
                onnx, prior_won = out / "run-best" / w, True
                choice = (f"best of the whole run: session {pbest['session']} of {sessions}, {pbest['ckpt']} at "
                          f"{pbest['run_steps']:,} run steps ({result_text(pr)}, rank key "
                          f"{[round(x, 4) for x in pr['rank_key']]})"
                          + (f"; this session's best {Path(best['onnx']).stem} {result_text(best)}" if best else ""))
        if resume and not prior_won and best is not None and Path(best["onnx"]) == onnx:
            name = (onnx.stem if onnx.name != "policy.onnx" else f"the final weights at {STEP_OFFSET + steps_done:,} run steps")
            choice = choice.replace("best by TS", f"best of the whole run by TS: session {sessions} of {sessions}, "
                                                  f"{name}" if pbest is not None else
                                    f"best of session {sessions} by TS (no earlier best known), {name}", 1)
        # the saved run of the exported weights' eval: train.final = true (a crash's partial export: train.ended "error" too)
        final_run = tse.mark_final(onnx, ended="error" if error is not None else None)
        if final_run:
            choice += f", run {final_run}"
        tse.close()
    ck_runs = tse.runs_summary() if tse is not None else None
    if ChallengeVecEnv.CURRENT is not None:  # what the rollout actually ran (the workers' reports), for the log and stats.json
        ro_desc = rollout.ran(ro, ChallengeVecEnv.CURRENT.pool)
        print(f"[duck-rl] rollout ran: {ro_desc}", flush=True)
    print(f"[duck-rl] submitting {onnx.name}: {choice}", flush=True)
    pt = next(out / "run-best" / n for n in pbest["files"] if n.endswith(".pt")) if prior_won else ckpt_of(onnx, out, steps_done)
    m = re.fullmatch(r"ckpt-(\d+)", onnx.stem)
    train_s = summary.get("train_s") or (hist[-1]["elapsed_s"] if hist else None)
    wall_s = round(time.time() - t_start, 1)
    # this step: the steps the exported weights were trained on (the selected checkpoint's), the run's training time
    # (what producing them cost, selection included), the run's steps and the command's wall time so far
    this_step = dict(steps=int(m.group(1)) - STEP_OFFSET if m else steps_done, train_s=train_s, run_steps=steps_done,
                     wall_s=wall_s)
    lin = moves.lineage(parent, this_step, moves.sha256_file(cfg_path), duck_rl.KIT_VERSION)
    if prior_won and pbest.get("lineage"):  # the exported weights are an earlier session's: their own lineage
        lin = pbest["lineage"]
    # kit 1.15.0: the run's best so far (a later --resume compares against it), its files relative to the run dir
    rb_rec = pbest["rec"] if prior_won else next((r for r in reversed(tse.records if tse is not None else [])
                                                 if r["onnx"] == str(onnx)), None)
    run_best = None
    if prior_won:
        run_best = {k: v for k, v in pbest.items() if k != "files"}
        run_best["files"] = [f"run-best/{n}" for n in pbest["files"]]
    elif rb_rec is not None and (m or pt is not None):
        run_best = dict(rec=rb_rec, session=sessions, ckpt=onnx.stem if m else pt.stem,
                        run_steps=int(m.group(1)) if m else STEP_OFFSET + steps_done, lineage=lin,
                        files=[onnx.name, (onnx.with_suffix(".pt") if m else pt).name])
    # model + minutes: the ranked caption "trained by <model> in N min" (this command's wall-clock plus the minutes of the
    # kit's own run it continues; compute inherited from library moves excluded; docs/ranked-strategies-decisions-log.md
    # RS-08)
    inherited = init["minutes"] if init is not None else 0.0  # the kit's own run continued (--init), not library moves
    if a.model is None:  # the agent's model, saved by the CLI (agent.json) while this trained
        a.model = saved_agent_model()
    # kit 1.8.1: the in-loop curve in one line (and its points, which a --resume of this run continues)
    start_rec = tse.start_rec if tse is not None else None
    if start_rec is None and resume and pbest is not None and rb_rec is not None and same_eval(pbest["rec"], rb_rec):
        start_rec = dict(pbest["rec"], start=f"the run's best before this session ({pbest['ckpt']})")
    curve = training_curve(tse.records if tse is not None else [], eval_info, train_s, prior, resume,
                           start=tse.start_rec if tse is not None else None)
    # N26: start -> trained on the same seeds and opponents (no re-run needed); N25: the busy machine, if it was
    svt = start_vs_trained(start_rec, rb_rec, eval_info, tse.cfg["seeds"] if tse is not None else None,
                           eval_opps, start_note if tse is not None else "the in-loop TS eval is off")
    busy_sum = watch.summary()
    meta = dict(**({"model": a.model} if a.model else {}), minutes=round(wall_s / 60.0 + inherited, 2),
                **({"device": dev} if dev else {}),
                **({"minutes_inherited": inherited} if inherited else {}),
                kit_version=duck_rl.KIT_VERSION, challenge=challenge, **({"eval_challenge": eval_challenge} if eval_challenge != challenge else {}),
                label=a.label, session=session, config=str(a.config), seed=a.seed, nodes=nodes,
                warm_start=warm_label, init=str(a.init) if a.init else None, steps=steps_done, iters=summary.get("iters"),
                train_s=summary.get("train_s"), stopped=stopped or summary.get("stopped_by"),
                sps=int(np.median([h["sps"] for h in hist])) if hist else None, onnx=str(onnx), choice=choice,
                ts_evals=tse.records if tse is not None else [], wall_s=wall_s, lineage=lin, host=hc["host"],
                **({"duel": dict(opponents=[[o["ref"], o["weight"]] for o in duel["opponents"]],
                                 eval_opponents=duel["eval_refs"] if eval_opps else [], unbound=duel["unbound"])} if duel else {}),
                **({"raw_command": hc["export"]} if hc["host"] == "raw" else {}),
                **({"command_ranges": hc["ranges"]} if hc["ranges"] else {}), **({"done": hc["done"]} if hc["done"] else {}),
                curve=curve, **({"run_best": run_best} if run_best else {}), **({"ended": "error", "error": error} if error is not None else {}))
    # kit 1.6.2: the run card ("time to target") in summary.json and stats.json, the learner's knobs and the recipe hash
    card = run_card(hist, tse.records if tse is not None else [], eval_info, getattr(cfg_mod, "TARGETS", None) or [],
                    seed=a.seed, recipe=recipe, knobs=knobs, overrides=overrides, train_s=summary.get("train_s"),
                    steps=steps_done)
    sp = out / "summary.json"
    if sp.exists():
        with contextlib.suppress(ValueError):
            sp.write_text(json.dumps(dict(json.loads(sp.read_text()), run_card=card, curve=curve, start_vs_trained=svt,
                                          busy=busy_sum), indent=1))
    (out / "stats.json").write_text(json.dumps(dict(meta, start_vs_trained=svt, busy=busy_sum, checkpoint_runs=ck_runs,
                                                    rollout=ro_desc, last=hist[-1] if hist else None,
                                                    learner=learner_record(a.learner, summary, hist, knobs=knobs, overrides=overrides,
                                                                           recipe=recipe, resume=resume), run_card=card), indent=1))
    print("[duck-rl] run card: " + card_line(card), flush=True)
    world = rb_rec.get("world") if rb_rec is not None else None
    # the library move the weights started from (not with --init: those weights start from the checkpoint)
    warm_ref = (dict(move=warm_mv.get("alias") or str(warm_mlp), pin=warm_mv["id"])
                if warm_mv is not None and init is None and moves.is_library_ref(str(warm_mlp)) else None)
    ck_all = checkpoints(out)  # the run's newest checkpoint (the final one; its Adam sidecar is the one kept): last.pt
    last_pt = ck_all[-1].with_suffix(".pt") if ck_all and ck_all[-1].with_suffix(".pt").exists() else None
    sub = export_submission(a.label, eval_info, program, nodes, onnx, HERE / "submissions" / a.label, meta, eval_challenge, ckpt=pt, world=world,
                            move_extra=hc["move_extra"], warm=warm_ref, last=last_pt, fixed_moves=fixed_moves, base=base)
    with contextlib.suppress(OSError):  # kit 1.22.0: the export's submission.json beside the logs too (the CLI uploads it with them)
        shutil.copyfile(sub / "submission.json", out / "submission.json")
    if ck_runs is not None and ck_runs["mode"] != "off":
        c = ck_runs
        print(f"[duck-rl] checkpoint runs: kept {c['kept']} of {c['evaluated']} ({mb(c['kept_mb'] * 1e6)} MB) in {c['dir']}"
              + (f"; all would have been ~{mb(c['all_mb_estimate'] * 1e6)} MB" if c["mode"] == "improved" else ""), flush=True)
    print(f"[duck-rl] done: {meta['steps']:,} env steps, median {meta['sps']} steps/s -> submission {sub}", flush=True)
    print(svt["line"], flush=True)
    if busy_sum is not None:
        print(f"[duck-rl] {busy_sum['line']}", flush=True)
    print(curve["line"], flush=True)
    # kit 1.16.1 (LB-207): "Training done" on the human's page, posted by the CLI itself (when logged in), so the agent
    # can't forget it; after everything else, never failing the run (bridge.post_done)
    # kit 1.26.0: a crash posts "Training crashed: <error>" the same way (and its logs go up)
    post_end(session, tse, error)
    if error is not None:
        print(f"[duck-rl] training had crashed: {error} (traceback above); exit 1", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    code, err, interrupted = 0, None, False
    try:
        code = main() or 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            err = e.code
    except KeyboardInterrupt:
        code, interrupted = 130, True
    except BaseException as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        code, err = 1, f"{type(e).__name__}: {e}"
    # kit 1.26.0: status.json's end (duck_rl/status.py): done, stopped (SIGTERM, Ctrl-C) or crashed with its error
    st = STATUS["s"]
    if st is None and DETACHED is not None:  # ended before run() made it: the parent's "running" record gets the end
        prev = status.read(DETACHED) or {}
        st = status.Status(DETACHED, prev.get("label") or Path(DETACHED).name, started_at=prev.get("started_at"))
    if st is not None:
        error = STATUS["error"] or err or (f"exit {code}" if code else None)
        st.end(status.end_state(code, error, interrupted or STOP["sig"] is not None), code, error)
    if DETACHED is not None:  # --detach: the log's last line says the training has ended
        print(f"[duck-rl] exit {code}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # as the lab's kit.py: skip C++ static destructors after pools / MLX / onnxruntime
