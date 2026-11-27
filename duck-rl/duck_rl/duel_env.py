"""The training env of a duel Challenge (PvP: Sumo): the learner's duck against a frozen opponent in duck_rl/duel.py's
DuelWorld, with ChallengeEnv's interface (duck_rl/env.py), so the pool (env.make_env), the learner and train.py run it as
they run a solo Challenge.

  sides     the learner = side A: the config's duckscript (PROGRAM) with its trained nodes (TRAINED_NODE) bound to an
            EXTERNAL move (the policy's action on every tick they run); its other nodes run as the sim runs them (the
            stock skills, e.g. get_up). The opponent = side B, drawn per episode from cfg opponents by weight (house
            bots, a duckscript, a duck-strategy/1, a kit submission: a frozen earlier policy), running inside the world;
  episodes  each reset draws an opponent, then `swapped` (probability cfg swap, 0.5: the learner in slot 1, the other
            spawn and the other random stream, as the server plays both sides of a seed), and takes a fresh training
            label "<challenge>:<version>:train:<env seed>:<n>" (env seed: ChallengePool's run seed x 100000 + worker x 1000 + env
            index; n: this env's episode count), so episodes differ per run seed, worker, env and episode and never
            replay a public ("<challenge>:<version>:<seed>") or hidden ("...:key:<hex>") seed. The match's own draws (spawn
            jitter, the spawn line's rotation, sole friction, trunk mass, sensor noise) are the training randomisation.
            The world then runs both sides' own runners to the learner's first tick (the start node is a trained one in
            the baseline: no wait);
  step      one learner action = one 50-Hz tick of both ducks (world.step). The episode ends, a termination (no
            bootstrap), on the decision (a win, a loss, a draw or invalid; the 20-s time-out is a decision too: the
            trunk closer to the centre wins) or on the row the learner's active node is not a trained node
            (`unbound`, e.g. recover after a fall: UNBOUND "end"; the TS eval and the server run those nodes as
            usual, so what follows still counts there, it is just not trained). UNBOUND "skip" (2026-10-12, for a
            policy on a subset of the walk nodes, e.g. only the home nodes): a fall still ends the episode as
            `unbound` (the learner left a trained node for a node whose skill is in skip_end_skills, default get_up),
            but any other untrained node is run inside the step by the duckscript's own moves (the stock skills)
            until the learner's next trained tick or the decision, and that row is the step's (one transition over
            the skipped ticks: their dense reward terms are not paid; the outcome is). "end" stays bit-identical. stagger_start (train.py
            STAGGER_START) cuts each env's first training episode at a random tick, a bootstrapped time-out. An
            invalid round (the physics broke: a non-finite state, or unstable at the decision) is one terminal
            transition that reads nothing of the broken MjData: its state row is the episode's last valid one (zeros
            on its first tick) with `terminated`, `invalid` and `decided` set, the obs and critic extras the last
            valid row's, so the reward only sees finite columns (the dense terms repeat the last valid row's);
  obs       the learner's own 61-dim policy obs (the TS runner's; nothing of the opponent: the duckscript's steering
            on target_* / goal_* reaches the policy through the command block);
  critic    the adapter's 6 extras (duck_rl/challenges/sumo.py priv: the learner's heading-frame velocity, both distances
            from the centre, the gap between the trunks, t / the duration);
  state     the lab's columns of the learner's duck (lab.fastenv.nb_state: heading and spawn frames, joints, actions,
            command, t; `fell` = the judge's raw fall state of the learner) + the adapter's (sumo.py: the events won /
            lost / draw / invalid / decided / unbound, distances, margins, speeds, tilts, contact);
  cost      ~0.3 CPU-ms per step, mostly mj_step of the two ducks (the opponent's policy runs inside: SimdMlp).
raw host (kit 1.10.0, from the 2026-10-15 night's f61d5443): duel host "raw" binds the trained nodes raw (raw_command
            {node: 13 floats}, the move's own block, e.g. steer-walk@1's speed / wz per node; head and body in the obs),
            and command_ranges {slot index: (lo, hi)} redraws those slots once per training episode for every trained
            node (DuelWorld.start_hook, before the first obs); without ranges each node keeps its block. The walk-stand
            host sets no hook (every other duel config is unchanged). The adapter may define on_reset(row) (per-episode
            state, e.g. Bread Fight's closing speed).
cfg (train.py duel_config; JSON-able): info, scene, duel {program (the learner's duckscript compiled by ts/prepare.ts
--program-full), nodes, unbound "end", opponents [{ref, spec, weight}] (spec: a compiled duckscript, or a
duck-strategy/1 dict whose program is compiled and whose move paths are absolute: opponent_spec), swap, host?,
raw_command?, command_ranges?}, stagger_start.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import bridge, moves
from .challenges import make_adapter
from .duel import DuelSide, DuelWorld, compile_program
from .env import FELL_I, TERM_I, columns, state_dim
from .program import EXTERNAL
from .prng import rng_key
from lab.duck import DT, yaw_of
from lab.fastenv import nb_state

UNBOUND_MODES = ("end", "skip")
SKIP_END_SKILLS = ("get_up",)  # UNBOUND "skip": leaving a trained node for a node running one of these ends the episode
END_REASONS = ("won", "draw", "lost", "invalid", "unbound", "timeout")  # DuelEnv.end_reason (train.py counts them)
HOUSE_TIERS = ("easy", "medium", "hard")  # the Arena bots (schema HOUSE_BOTS); each Challenge ships its own (Sumo, Wobbly Raft: no hard)
MAX_STARTS = 20  # resets in a row in which the learner never got a tick before the decision: a config error
MAX_WEIGHT = 1e6  # an OPPONENT weight's bound (parse_opponents): their sum stays finite


class DuelEnv:
    PRIV = 6

    def __init__(self, cfg: dict, seed: int, fused: bool = False):
        self.cfg = cfg
        self.info = info = cfg["info"]
        dc = cfg["duel"]
        self.challenge_id, self.version = info["challenge"]["id"], int(info["challenge"]["version"])
        self.rng_key = rng_key(info["challenge"])  # the PRNG labels' name (LB-370: a renamed Challenge's old id)
        self.unbound = dc.get("unbound", "end")
        if self.unbound not in UNBOUND_MODES:
            raise ValueError(f"unbound {self.unbound!r}: {', '.join(map(repr, UNBOUND_MODES))} (v1)")
        self.world = w = DuelWorld(challenge=self.challenge_id)
        if fused:  # the fast rollout (duck_rl/fused.py): the same physics, in one nogil call per tick
            w.use_fused()
        self.obs_dim = 61  # env.py's workers size the obs batch by it (kit 1.16.0); a duel learner has no eye
        self.nodes = list(dc["nodes"])
        ends = set(dc.get("skip_end_skills", SKIP_END_SKILLS))
        self.skip_end = {nid for nid, n in dc["program"]["nodes"].items() if n.get("skill") in ends}  # (UNBOUND "skip")
        self.skipped = 0  # UNBOUND "skip": ticks the world ran without the learner in the last step (0 in "end")
        entry = dict(nodes=self.nodes, move=EXTERNAL)
        self.host = dc.get("host", "walk-stand")
        if self.host == "raw":  # the raw host (module doc): each node's NodeMove.command, redrawn per episode on ranges
            entry.update(host="raw", command={n: [float(x) for x in dc["raw_command"][n]] for n in self.nodes})
        w.set_side(0, dc["program"], [entry, *[dict(e) for e in dc.get("base_moves") or []]], name="learner")  # + BASE_STRATEGY
        self._lmoves = [w.sides[0].moves[n] for n in self.nodes]
        self._base_cmd = {n: np.asarray(dc["raw_command"][n], float) for n in self.nodes} if self.host == "raw" else {}
        self.ranges = {int(i): (float(r[0]), float(r[1])) for i, r in (dc.get("command_ranges") or {}).items()}
        if self.host == "raw" and self.ranges:
            w.start_hook = self._draw_command
        self.opponents = [DuelSide(self.challenge_id, w.rt, o["spec"], name=o["ref"]) for o in dc["opponents"]]
        self.refs = [o["ref"] for o in dc["opponents"]]
        wt = np.array([float(o.get("weight", 1.0)) for o in dc["opponents"]])
        self.cum = np.cumsum(wt / wt.sum())
        self.swap = float(dc.get("swap", 0.5))
        self.rng = np.random.default_rng(seed)
        self.seed, self.n = int(seed), 0
        self.challenge_s = float(info["duration_s"])
        self.ex, self.dim = columns(info), state_dim(info)
        self.adapter = make_adapter(self)
        self.end_s = self.episode_s = float(self.adapter.default_end_s())
        self.priv_dim = self.PRIV
        self.stagger, self._first_train_reset, self._cut_k = bool(cfg.get("stagger_start", False)), True, None
        # each slot's views into the shared MjData (the learner's slot changes with `swapped`): nb_state reads one duck
        d = w.d
        self._qv = [d.qpos[k.root_qpos_adr:] for k in w.ducks]
        self._vv = [d.qvel[k.root_dof_adr:] for k in w.ducks]
        self._qrel = [k.qadr - k.root_qpos_adr for k in w.ducks]
        self._vrel = [k.vadr - k.root_dof_adr for k in w.ducks]
        self._soles = [np.asarray(k.soles, np.int64) for k in w.ducks]
        self._trk = np.zeros(10)
        self._pr = np.zeros(self.PRIV, np.float32)
        self._last = np.zeros(14, np.float32)
        self._st_ok = np.zeros(self.dim, np.float32)  # the episode's last valid state row (an invalid round's: step)
        self.cur, self.swapped, self.label, self.row = None, False, None, None
        self.k, self.score, self.end_reason, self.unbound_now, self.from_bank = 0, 0.0, None, False, False

    # ---- resets ----
    def reset(self, ep=None, cmd=None):
        """A training episode (module doc): the opponent, the side, the label; the world runs to the learner's first
        tick. Returns (obs, priv). `ep` / `cmd` (ChallengeEnv's explicit draws) do not apply to a match."""
        w = self.world
        for _ in range(MAX_STARTS):
            i = min(int(np.searchsorted(self.cum, self.rng.random(), side="right")), len(self.cum) - 1)
            self.swapped = bool(self.rng.random() < self.swap)
            if i != self.cur:
                w.set_side(1, self.opponents[i])
                self.cur = i
            self.label = f"{self.rng_key}:{self.version}:train:{self.seed}:{self.n}"
            self.n += 1
            w.reset(self.label, self.swapped)
            row = w.advance(0)
            if row.external[0] and not row.decided:
                break
        else:
            raise RuntimeError(f"{MAX_STARTS} matches in a row were decided before the learner's nodes {self.nodes} ran "
                               "once: TRAINED_NODE should hold the nodes the duck plays the match in (the start node)")
        self.row, self.k = row, row.k
        slot = w.slot_of[0]
        q = self._qv[slot]
        yaw = yaw_of(q[3:7])
        self._trk[:] = [q[0], q[1], yaw, yaw, 0.0, 0.0, 0.0, 0.0, q[2], 0.0]
        self._last[:] = w._prev[slot]  # the learner duck's last action (zeros at tick 0; the stock's after a wait)
        self._st_ok[:] = 0.0
        self.score, self.end_reason, self.unbound_now = 0.0, None, False
        self._cut_k = None
        if hasattr(self.adapter, "on_reset"):
            self.adapter.on_reset(row)
        if self._first_train_reset:
            self._first_train_reset = False
            n_left = int(round(self.end_s / DT)) - self.k
            if self.stagger and n_left > 1:  # STAGGER_START: the first episode ends at a random tick
                self._cut_k = self.k + int(self.rng.integers(1, n_left + 1))
        return row.obs[0].copy(), self._priv(row)

    def _draw_command(self, _w) -> None:
        """DuelWorld.start_hook of the raw host with command_ranges (module doc): one draw per ranged slot for this
        episode, written into every trained node's command block (its own base block otherwise)."""
        draw = {i: (self.rng.uniform(lo, hi) if hi > lo else lo) for i, (lo, hi) in self.ranges.items()}
        for n, mv in zip(self.nodes, self._lmoves):
            c = self._base_cmd[n].copy()
            for i, x in draw.items():
                c[i] = x
            mv.command[:] = c

    def _priv(self, row) -> np.ndarray:
        self.adapter.priv(row, self._pr)
        return self._pr.copy()

    # ---- one learner tick ----
    def step(self, action):
        w = self.world
        a = np.asarray(action, np.float32)
        row = self.row = w.step({0: a})
        self.k = row.k
        decided = bool(row.decided or row.stopped)
        self.skipped = 0
        if (self.unbound == "skip" and not decided and not row.external[0] and not row.invalid
                and row.node[0] not in self.skip_end):  # an untrained, non-fall node: the world runs it (module doc)
            k0 = row.k
            row = self.row = w.advance(0)
            self.k, self.skipped = row.k, row.k - k0
            decided = bool(row.decided or row.stopped)
        self.unbound_now = unbound = not decided and not row.external[0]
        term = decided or unbound
        timeout = not term and self._cut_k is not None and self.k >= self._cut_k
        if row.invalid:  # the physics broke: nothing of the MjData is read (module doc: step), so no NaN gets out
            st = self._st_ok.copy()
            st[TERM_I] = 1.0
            self.adapter.fill_events(st, self.ex, row)
            pr = self._pr.copy()  # row.obs[0] is the last valid row's too (DuelWorld observes nothing of a broken row)
        else:
            slot = w.slot_of[0]
            x, d = w.ducks[slot], w.d
            st = np.empty(self.dim, np.float32)
            nb_state(st, self._trk, d.xmat[x.trunk], self._qv[slot], self._vv[slot], d.sensordata, x.gyro_adr,
                     d.xpos[x.head_body, 2], d.geom_xpos, self._soles[slot], d.contact.geom, d.ncon,
                     w.bams[slot].prev_torque, self._qrel[slot], self._vrel[slot], x.default_pos, a, self._last,
                     w.programs[slot].command, self.k, DT, self.challenge_s, float(row.down[0]), float(term), 0.0)
            self.adapter.fill(st, self.ex, row)
            self._st_ok[:] = st
            pr = self._priv(row)
        self._last[:] = a if not self.skipped else w._prev[w.slot_of[0]]  # skip: the duck's own last action
        if decided:
            self.score = float(row.points[0])
            self.end_reason = ("invalid" if row.invalid else "won" if row.winner == 0 else "lost"
                               if row.winner == 1 or row.reason == "timeout_both_lose" else "draw")  # both lose: a loss
        else:
            self.end_reason = "unbound" if unbound else "timeout" if timeout else None
        return row.obs[0].copy(), pr, st, term, timeout

    @property
    def disp_now(self) -> float:
        """|displacement from the episode's first learner row| (m)."""
        return float(np.hypot(self._trk[5], self._trk[6]))

    def episode_stats(self, s: np.ndarray, term: bool) -> dict:
        """The pool's record of the episode that just ended on state row s: ChallengeEnv's fields in their duel meanings
        (lost = a loss, success = a win, score = the learner's points, 0 when unbound) plus the outcome."""
        row = self.row
        return dict(len=self.k, fell=float(s[FELL_I] > 0), terminated=float(term), disp=self.disp_now,
                    lost=float(self.end_reason == "lost"), score=self.score, success=float(self.end_reason == "won"),
                    t_end=self.k * DT, bank=0.0, reason=self.end_reason, opponent=self.refs[self.cur],
                    swapped=self.swapped, match_reason=row.reason if row.decided else None)


# ---- the opponents (resolved in the parent: the workers get compiled specs) ----
def house_bots_root() -> Path:
    """apps/cli/src/match.ts houseBotsRoot: DUCKBENCH_HOUSE_BOTS_DIR, else the repo's packages/challenges/baselines, else the
    player package's baselines/ (beside the CLI's dist/)."""
    if os.environ.get("DUCKBENCH_HOUSE_BOTS_DIR"):
        return Path(os.environ["DUCKBENCH_HOUSE_BOTS_DIR"]).expanduser().resolve()
    if bridge.REPO is not None:
        return bridge.REPO / "packages" / "challenges" / "baselines"
    return Path(bridge.CLI[1]).resolve().parent.parent / "baselines"


def builtin_opponent(ref: str, challenge: str) -> tuple[str, Path] | None:
    """A built-in opponent ref, "house:<tier>" or "stand", as the TS eval's --opponent takes it: (the ref normalised,
    its file), else None (a path). SystemExit for a house tier or a stand-still baseline the Challenge does not ship."""
    s = ref.strip()
    root = house_bots_root() / challenge
    if s.lower().startswith("house:"):
        tier = s[6:].strip().lower()
        f = root / f"house-{tier}.json"
        if tier not in HOUSE_TIERS or not f.exists():
            have = sorted(p.stem.replace("house-", "house:") for p in root.glob("house-*.json"))
            raise SystemExit(f"opponent {ref!r}: no such house bot for {challenge} ({f}); there: {', '.join(have) or 'none'}")
        return f"house:{tier}", f
    if s == "stand":
        f = root / "stand.json"
        if not f.exists():
            raise SystemExit(f"opponent 'stand': {challenge} has no stand-still baseline ({f})")
        return "stand", f
    return None


def opponent_spec(path: Path, challenge: str) -> dict:
    """An opponent file as DuelSide takes it in a worker, made here once: a duckscript (duck-program/1 or /2) compiled (ts/prepare.ts
    --program-full, one node process), or a duck-strategy/1 (a submission dir: its submission.json) with its duckscript
    compiled and every move's path absolute (a library ref resolved to its file, a raw card's knobs attached).
    ValueError for anything else."""
    f = path / "submission.json" if path.is_dir() else path
    raw = json.loads(f.read_text())
    fmt = raw.get("format") if isinstance(raw, dict) else None
    if fmt in ("duck-program/1", "duck-program/2"):
        return compile_program(challenge, f)
    if fmt != moves.STRATEGY_FORMAT:
        raise ValueError(f"{f}: format {fmt!r}; an opponent is a duck-program/1 or /2 or {moves.STRATEGY_FORMAT} file, "
                         "or a submission dir")
    other = (raw.get("challenge") or raw.get("dare") or {}).get("id")  # a pre-rename envelope: dare  rename-challenge:keep
    if other not in (None, challenge):
        raise ValueError(f"{f}: a strategy for {other}, not {challenge}")
    prog = raw.get("program")
    prog = compile_program(challenge, prog, f.parent) if isinstance(prog, str) else compile_program(challenge, prog)
    entries = []
    for e in raw.get("moves") or []:
        e = {k: v for k, v in e.items() if k in ("nodes", "move", "pin", "host", "command", "done")}
        ref = str(e.get("move"))
        if moves.is_library_ref(ref):
            r = moves.resolve_move(ref)
            e["move"] = r["json"]
            if e.get("host") == "raw":
                e["knobs"] = moves.knobs_of(ref, r)
        else:
            p = Path(ref).expanduser()
            p = p if p.is_absolute() else (f.parent / p).resolve()
            # an Arena bot's move in the player package ships without its .bin: the CLI's copy (fetched on first use)
            e["move"] = moves.resolve_path(p)["json"] if p.exists() and not moves.bin_of(p).exists() else str(p)
        entries.append(e)
    return {"format": moves.STRATEGY_FORMAT, "program": prog, "moves": entries}


def parse_opponents(value) -> list[tuple[str, float]]:
    """OPPONENT -> [(ref, weight)]: a ref, a list of refs (uniform), or a list of (ref, weight) pairs. SystemExit on a
    malformed one, a weight that is not a finite number in (0, MAX_WEIGHT] (DuelEnv normalises their sum) or a ref given
    twice."""
    items = [value] if isinstance(value, str) else list(value or [])
    out: list[tuple[str, float]] = []
    for it in items:
        if isinstance(it, str):
            ref, wt = it, 1.0
        elif isinstance(it, (list, tuple)) and len(it) == 2 and isinstance(it[0], str):
            ref, wt = it[0], it[1]
        else:
            raise SystemExit(f"OPPONENT entry {it!r}: a ref (house:easy|medium|hard, stand, a file or a submission "
                             "dir) or a (ref, weight) pair")
        if isinstance(wt, bool) or not isinstance(wt, (int, float)) or not 0 < wt <= MAX_WEIGHT:  # NaN fails too
            raise SystemExit(f"OPPONENT {ref!r}: weight {wt!r}, want a number > 0 and <= {MAX_WEIGHT:g} (only the ratios "
                             "count)")
        out.append((ref.strip(), float(wt)))
    if not out:
        raise SystemExit("OPPONENT: no opponent")
    seen = [r for r, _ in out]
    dup = next((r for i, r in enumerate(seen) if r in seen[:i]), None)
    if dup is not None:
        raise SystemExit(f"OPPONENT {dup!r} is given twice (give it a weight instead)")
    return out
