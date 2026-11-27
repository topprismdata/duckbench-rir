"""The duckscript runtime in Python, one instance per duck of a match (duck_rl/duel.py): the TS program interpreter of
packages/sim/src/runtime/match.ts MatchDuck (skills layer, sensors, fall events, program ticks at 10 Hz), mirrored
operation for operation for the skills a Sumo duckscript runs, so a Python match replays a TS one.

  program    the compiled duckscript (ts/prepare.ts --program-full: schema CompiledProgram, nodes in the sim's node
             order, guard ASTs): CompiledProgram; guards are evaluated as schema evaluateGuard, transitions as
             nextTransition (the first true `until` rule, else max_time once t_state >= s);
  skills     walk, stand, get_up (runtime/skills.ts SkillLayer: the walk controller's speed map, start kick when
             entered from a stop and heading hold reset on every enter; get_up done = upright 1 s or give up after
             6 s, then the episodic fallback to stand; `fallen` suppressed while get_up runs; END = stand). The lead-ins
             of sit / stand_up / kicks / roulade are not ported: a program with another skill is refused (ValueError);
  signals    every one (runtime/sensors.ts): t, t_state, tilt, upright, fallen, done, speed, yaw_rate, height (from the
             stage top), obstacle_dist, target_* (the opponent's head body), goal_* (the Challenge's goal point), progress
             (0 in a match, as runMatch feeds it); the two noise values of each target_* / goal_* look are drawn from
             the duck's own stream (prng.py) in TS's order: target first, then goal, every program tick; kit 1.9.0
             (LB-108): rival_bearing / rival_dist / role (the other duck's trunk, all around, no noise; role = the
             slot), hill_bearing / hill_dist (the centre of a raft, sweeper or log prop: match.ts raftAt; no such
             prop: 0 / 5), log_off / log_spin (Log Roll), goal_* all around from the trunk with duel.chase.around;
             kit 1.10.0 (Bread Fight): crumb_bearing / crumb_dist / crumb_rival_dist (the crumb nearest to the duck),
             crumb2_* (the second-nearest), crumbs, rival_crumbs, from the match's crumb field (duck_rl/crumbs.py
             CrumbSense; none out: 0 / 5 / 5; no field: also crumbs 0, rival_crumbs 0, as sensors.ts);
             any other signal is refused (ValueError);
  moves      per node (walk or stand nodes, schema StrategyMove / sim NodePolicy): a trained move (a SimdMlp: the
             walk-stand host, the stock command block; or the raw host, its own 13-value block, head and body in the
             obs) or EXTERNAL (the caller supplies the action on the ticks the node runs: a training env's learner);
             its own done rule, counted in its ticks (skills.ts moveDoneTicks).
Per policy tick k the caller (duel.py) runs observe_row (the row's posture), then step(k, data): the skills layer's
observe (k > 0), the program tick every PROGRAM_EVERY ticks, the layer's step; afterwards the tick's runner (or
EXTERNAL), command block and obs masks are on the instance (runner, command, head_in_obs, body_in_obs, external).
The 61-dim obs (head EMA, lagged joint velocities, prev action) is the world's (lab.fastenv.nb_obs), as in TS DuckLoop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .controller import (KICK_COMMAND, KICK_TICKS, LATERAL_GAIN, LATERAL_MAX, POLICY_DT, WALK_BACKWARD, WALK_FORWARD,
                         WZ_LIMIT, YAW_GAIN, _clip, walk_command_vx)
from .jsmath import js_fixed, js_hypot2, js_round
from . import v2 as V2

EXTERNAL = "external"  # a move entry's `move` (or NodeMove.runner): the caller supplies the action on the node's ticks
SUPPORTED_SKILLS = ("walk", "stand", "get_up", "sit", "stand_up", "kick_left", "kick_right", "ground_pick", "roulade")
HOST_SKILLS = ("walk", "stand")  # node-policies.ts HOST_SKILLS: the nodes a trained move can run on
DONE_KEYS = ("after_s", "upright_s", "rolled_then_upright_s")  # schema MoveDone
TARGET_SIGNALS = ("target_seen", "target_bearing", "target_dist")
GOAL_SIGNALS = ("goal_seen", "goal_bearing", "goal_dist")
RIVAL_SIGNALS = ("rival_bearing", "rival_dist", "role")  # sensors.ts options.rival (1v1: role = the slot)
HILL_SIGNALS = ("hill_bearing", "hill_dist")  # sensors.ts options.hill: here only match.ts raftAt (raft, sweeper, log)
LOG_SIGNALS = ("log_off", "log_spin")  # Log Roll (sensors.ts)
CRUMB_SIGNALS = ("crumb_bearing", "crumb_dist", "crumb_rival_dist", "crumbs", "crumb2_bearing", "crumb2_dist",
                 "crumb2_rival_dist", "rival_crumbs")  # Bread Fight (sensors.ts options.crumbs)
HILL_PROPS = ("raft", "sweeper", "log")  # match.ts raftAt's kinds the Python match runs (gust_pier, roomba: refused)
KNOWN_SIGNALS = ("t", "t_state", "tilt", "upright", "fallen", "done", "speed", "yaw_rate", "height", "progress",
                 "obstacle_dist", *TARGET_SIGNALS, *GOAL_SIGNALS, "heading", "forward_speed", "side_speed", "touching",
                 "touch_bearing", *RIVAL_SIGNALS, *HILL_SIGNALS, *LOG_SIGNALS, *CRUMB_SIGNALS)
TOUCH_SIGNALS = ("touching", "touch_bearing")


def wrap_pi(b: float) -> float:
    """b - 2 pi round(b / 2 pi) (sensors.ts, Math.round)."""
    return b - 2 * math.pi * js_round(b / (2 * math.pi))


class MatchRuntime:
    """The runtime constants of a duel Challenge's info (ts/prepare.ts `duel.runtime`, read from runtime/constants.ts and
    the schema): ticks, posture thresholds, sensors, randomisation, the skills' kinds, done rules (in ticks) and
    policies. The walk controller's constants are controller.py's (checked against the info: TS changed = an error)."""

    def __init__(self, info: dict):
        # a duel Challenge's info carries it under duel.runtime; a solo Challenge under program_runtime (every Challenge: the solo env's
        # program tick, feed and head holds) and, with a ported mechanic, also at the top level (ts/prepare.ts
        # soloMechanics; the solo program runtime, env.py program mode): the same constants
        rt = info["duel"]["runtime"] if "duel" in info else (info.get("runtime") or info.get("program_runtime") or info)
        self.raw = rt
        self.ticks_per_s = int(rt["ticks_per_s"])
        self.policy_dt = float(rt["policy_dt"])
        self.program_every = int(rt["program_every"])
        self.program_hz = self.ticks_per_s / self.program_every  # 10: sensors.ts t = n / 10, speed over 0.1 s
        self.max_ticks = int(rt["max_ticks"])
        self.cos_upright = float(rt["cos_upright"])
        self.upright_tilt_deg = float(rt["upright_tilt_deg"])
        # the duel pieces (kit 1.9.0): a solo Challenge's info has no duel block, so no hill, log or chase there (LB-123)
        duel = info.get("duel") or {}
        scene = duel.get("scene") or []  # ts/prepare.ts duel.scene (kit 1.9.0; absent in an older export)
        hill = next((p for p in scene if p["kind"] in HILL_PROPS), None)
        self.hill_pad = None if hill is None else (float(hill["x"]), float(hill["y"]))
        self.goal_around = bool(((duel.get("rules") or {}).get("chase") or {}).get("around"))
        log = next((p for p in scene if p["kind"] == "log"), None)
        self.log = None if log is None else (str(log["id"]), float(log["y"]))
        self.fallen_tilt_deg = float(rt["fallen_tilt_deg"])
        self.fallen_ticks = int(rt["fallen_ticks"])
        self.view_max_dist = float(rt["view_max_dist"])
        self.cos_view = float(rt["cos_view"])
        self.los_tol = float(rt["line_of_sight_tolerance"])
        self.ray_max = float(rt["ray_max"])
        self.noise_bearing = float(rt["noise_bearing"])
        self.noise_dist = float(rt["noise_dist_fraction"])
        self.groups = np.zeros(6, np.uint8)
        self.groups[int(rt["prop_geom_group"])] = 1  # sensors.ts GROUPS: rays see prop geoms only
        self.dr = rt["dr"]
        self.head_ema = float(rt["head_ema"])
        wp = rt.get("walk_params") or {"vx": {"min": -0.16, "max": 0.19}, "vy": {"min": -0.06, "max": 0.06},
                                        "wz": {"min": -0.6, "max": 0.6}}
        self.walk_ranges = [(float(wp[k]["min"]), float(wp[k]["max"])) for k in ("vx", "vy", "wz")]
        # skills.ts lead-ins (constants.ts); an older info lacks them: the TS values of sim 0.4.0
        self.sit_exit_ticks = int(rt.get("sit_exit_ticks", 25))
        rs = rt.get("roulade_settle", {"speed": 0.03, "ticks": 15, "maxTicks": 50})
        self.settle_speed, self.settle_ticks, self.settle_max = float(rs["speed"]), int(rs["ticks"]), int(rs["maxTicks"])
        kh = rt.get("kick_head_reset", {"tolerance": 0.15, "maxTicks": 25})
        self.kick_head_tol, self.kick_head_max = float(kh["tolerance"]), int(kh["maxTicks"])
        self.ground_pick_phase = float(rt.get("ground_pick_phase_per_tick", 0.005))
        self.head_joints = [int(j) for j in rt.get("head_joints", [6, 7, 8])]
        self.end = rt["end"]
        self.skill_names = list(rt["skill_names"])
        self.skill_index = {s: i for i, s in enumerate(self.skill_names)}
        self.kind = {s: v["kind"] for s, v in rt["skills"].items()}
        self.policy = {s: v["policy"] for s, v in rt["skills"].items()}
        self.stock_done = {s: self._done_ticks(v["done_rule"]) for s, v in rt["skills"].items()}
        w = rt["walk"]
        mine = dict(forward=[list(p) for p in WALK_FORWARD], backward=[list(p) for p in WALK_BACKWARD],
                    lateralGain=LATERAL_GAIN, lateralMaxCommand=LATERAL_MAX, kickCommand=KICK_COMMAND,
                    kickTicks=KICK_TICKS, yawGain=YAW_GAIN, wzCommandLimit=WZ_LIMIT)
        theirs = {k: w[k] for k in mine}
        if mine != theirs or self.policy_dt != POLICY_DT:
            raise ValueError(f"duck_rl/controller.py's walk constants differ from the TS sim's (constants.ts WALK): "
                             f"{mine} vs {theirs}; update controller.py")

    def ticks(self, s: float) -> int:
        """skills.ts ticks: Math.round(s / POLICY_DT)."""
        return js_round(s / self.policy_dt)

    def _done_ticks(self, rule: dict) -> dict | None:
        """skills.ts stockDoneTicks: a catalogue skill's done rule in ticks (None = never)."""
        k = rule["kind"]
        if k == "never":
            return None
        if k == "after":
            return dict(after=self.ticks(rule["s"]), upright=None, rolled=None)
        if k == "upright_for":
            return dict(after=self.ticks(rule["give_up_s"]), upright=self.ticks(rule["s"]), rolled=None)
        if k == "rolled":
            return dict(after=self.ticks(rule["max_s"]), upright=None, rolled=self.ticks(rule["s"]))
        raise ValueError(f"unknown done rule {rule!r}")

    def move_done_ticks(self, done: dict | None) -> dict | None:
        """skills.ts moveDoneTicks: a trained move's MoveDone (seconds) in ticks."""
        if done is None:
            return None
        bad = [k for k in done if k not in DONE_KEYS]
        if bad or not done or any(not (float(v) > 0) for v in done.values()):
            raise ValueError(f"done {done!r}: 1-3 of {', '.join(DONE_KEYS)}, each > 0 s")
        t = lambda k: None if done.get(k) is None else self.ticks(float(done[k]))  # noqa: E731
        return dict(after=t("after_s"), upright=t("upright_s"), rolled=t("rolled_then_upright_s"))


# ---- the compiled program --------------------------------------------------------------------------------------------
def compile_guard(g: dict):
    """A guard AST (schema Guard) -> fn(signals dict) -> bool, as schema evaluateGuard (or: first true; and: all)."""
    kind = g["kind"]
    if kind == "or":
        args = [compile_guard(a) for a in g["args"]]
        return lambda s: any(f(s) for f in args)
    if kind == "and":
        args = [compile_guard(a) for a in g["args"]]
        return lambda s: all(f(s) for f in args)
    if kind == "not":
        f = compile_guard(g["arg"])
        return lambda s: not f(s)
    if kind == "bool":
        name = g["signal"]
        return lambda s: s[name]
    if kind == "cmp":
        name, op, v = g["signal"], g["op"], float(g["value"])
        if op == "<":
            return lambda s: s[name] < v
        if op == "<=":
            return lambda s: s[name] <= v
        if op == ">":
            return lambda s: s[name] > v
        if op == ">=":
            return lambda s: s[name] >= v
        if op == "==":
            return lambda s: s[name] == v
        if op == "!=":
            return lambda s: s[name] != v
        raise ValueError(f"guard op {op!r}")
    raise ValueError(f"guard kind {kind!r}")


def guard_atoms(g: dict) -> list[dict]:
    """schema guardAtoms: the comparisons and boolean atoms, left to right."""
    if g["kind"] in ("or", "and"):
        return [a for x in g["args"] for a in guard_atoms(x)]
    if g["kind"] == "not":
        return guard_atoms(g["arg"])
    return [g]


@dataclass
class Node:
    id: str
    skill: str
    params: dict
    until: list  # [(fn, go, text)]: fn(signals) for duck-program/1, fn(v2.Ctx) for duck-program/2
    max_time: tuple | None  # (s, go)
    index: int  # in the sim's node order (frame node_index)
    set: tuple = ()  # duck-program/2: [(var, fn)], applied on entry
    feed: tuple = ()  # duck-program/2: [(command slot index, fn)]
    head: tuple = ()  # duck-program/2: [(head joint index 0-3, fn)]


class CompiledProgram:
    """A duckscript as ts/prepare.ts --program-full writes it (schema CompiledProgram), ready to run: nodes in the sim's
    node order (runtime/episode.ts nodeOrder: the compiled key order), guards compiled, the signals the guards read.
    ValueError for a skill or signal this runtime cannot run."""

    def __init__(self, compiled: dict, name: str = "program", extra_signals: tuple = ()):
        """extra_signals: signals a solo Challenge's runtime adds (SoloSensors: its movers' senses, maze_yaw)."""
        known = self.known = (*KNOWN_SIGNALS, *extra_signals)
        fmt = compiled.get("format")
        if fmt not in ("duck-program/1", "duck-program/2") or not isinstance(compiled.get("nodes"), dict):
            raise ValueError(f"{name}: not a compiled duckscript (duck-program/1 or /2): compile it first with "
                             f"duck_rl.duel.compile_program (the TS validator), or give the kit the program's .json file")
        self.name, self.raw = name, compiled
        self.start = compiled["start"]
        self.order = list(compiled["nodes"])
        self.nodes: dict[str, Node] = {}
        self.signals: set[str] = set()
        self.v2 = fmt == "duck-program/2"
        if self.v2:
            self._compile_v2(compiled, name)
            return
        for i, (nid, n) in enumerate(compiled["nodes"].items()):
            if "maxTime" not in n or any("guard" not in t for t in n.get("until", [])):
                raise ValueError(f"{name}: node {nid!r} is not compiled (ts/prepare.ts --program-full writes guards and maxTime)")
            if n["skill"] not in SUPPORTED_SKILLS:
                raise ValueError(f"{name}: node {nid!r} runs skill {n['skill']!r}; the Python match runtime runs only "
                                 f"{', '.join(SUPPORTED_SKILLS)} (duck_rl/program.py)")
            until = []
            for t in n["until"]:
                for a in guard_atoms(t["guard"]):
                    if a["signal"] not in known:
                        raise ValueError(f"{name}: node {nid!r} reads signal {a['signal']!r}, which the Python match "
                                         f"runtime does not compute")
                    self.signals.add(a["signal"])
                until.append((compile_guard(t["guard"]), t["go"], t["if"]))
            mt = n["maxTime"]
            self.nodes[nid] = Node(nid, n["skill"], dict(n["params"]), until,
                                   None if mt is None else (float(mt["s"]), mt["go"]), i)

    def _compile_v2(self, compiled: dict, name: str) -> None:
        """duck-program/2 (v2.py): expressions compiled to fn(Ctx); lets in order; vars; any rules; memory calls."""
        sig: set[str] = set()
        self.lets = {}
        for ln, e in compiled.get("lets", {}).items():
            self.lets[ln] = V2.compile_expr(e)
            V2.expr_signals(e, sig)
        self.var_init = {k: float(v) for k, v in compiled.get("vars", {}).items()}
        self.any = []
        for r in compiled.get("any", []):
            self.any.append((V2.compile_expr(r["expr"]), r["go"], tuple(r.get("except", ()))))
            V2.expr_signals(r["expr"], sig)
        self.handlers = {r["go"] for r in compiled.get("any", [])}
        self.memory_calls = list(compiled.get("memory", []))
        self.uses_routes = '"fn": "path' in __import__("json").dumps(compiled)
        for i, (nid, n) in enumerate(compiled["nodes"].items()):
            if n["skill"] not in SUPPORTED_SKILLS:
                raise ValueError(f"{name}: node {nid!r} runs skill {n['skill']!r}; the kit's match runtime runs only "
                                 f"{', '.join(SUPPORTED_SKILLS)} nodes (duck_rl/program.py); the TS eval runs every skill")
            until = []
            for t in n.get("until", []):
                until.append((V2.compile_expr(t["expr"]), t["go"], t["if"]))
                V2.expr_signals(t["expr"], sig)
            sets = tuple((a["name"], V2.compile_expr(a["expr"])) for a in n.get("set", []))
            feed = tuple((int(f["index"]), V2.compile_expr(f["expr"])) for f in n.get("feed", []))
            head = tuple((int(h["index"]), V2.compile_expr(h["expr"])) for h in n.get("head", []))
            for group in (n.get("set", []), n.get("feed", []), n.get("head", [])):
                for a in group:
                    V2.expr_signals(a["expr"], sig)
            mt = n.get("maxTime")
            self.nodes[nid] = Node(nid, n["skill"], dict(n["params"]), until,
                                   None if mt is None else (float(mt["s"]), mt["go"]), i, sets, feed, head)
        bad = sorted(x for x in sig if x not in self.known)
        if bad:
            raise ValueError(f"{name}: reads signal(s) {', '.join(bad)}, which the kit's match runtime does not compute")
        self.signals = sig

    @staticmethod
    def next_transition(node: Node, s: dict):
        """schema nextTransition: (go, via, rule) of the first true until rule, else max_time once t_state >= s; None = stay."""
        for rule, (f, go, _) in enumerate(node.until):
            if f(s):
                return go, "until", rule
        if node.max_time is not None and s["t_state"] >= node.max_time[0]:
            return node.max_time[1], "max_time", None
        return None


@dataclass
class NodeMove:
    """A move bound to a walk/stand node (skills.ts NodeMove): `runner` (a SimdMlp, or EXTERNAL), `command` (the raw
    host's 13-value block; None = the walk-stand host), `done` (ticks, MatchRuntime.move_done_ticks; None = never)."""
    runner: object
    command: np.ndarray | None = None
    done: dict | None = None
    name: str = "move"


def bind_moves(program: CompiledProgram, moves: list[dict] | None, rt: MatchRuntime, load) -> dict[str, NodeMove]:
    """node-policies.ts validateNodePolicies + nodeMoves for duck-strategy/1 `moves` entries {nodes, move, host?,
    command?, done?}: node id -> NodeMove. `move` is EXTERNAL, a loaded runner (anything with run(obs, out)), or a ref
    that `load(ref)` turns into one. Raw commands: 13 numbers or {slot name: value} (controller.command_vector)."""
    from .controller import command_vector
    out: dict[str, NodeMove] = {}
    raw_nodes = {n for m in (moves or []) if (m.get("host") or "walk-stand") == "raw" for n in (m.get("nodes") or ())}
    for nid, node in program.nodes.items():  # node-policies.ts checkFeeds
        if not node.feed or nid in raw_nodes:
            continue
        if node.skill != "walk":
            raise ValueError(f'node "{nid}" has a feed but no raw trained move: a {node.skill} node feeds only a raw-host move')
        if any(i > 2 for i, _ in node.feed):
            raise ValueError(f'node "{nid}" feeds a slot other than vx, vy, wz without a raw trained move')
    for i, m in enumerate(moves or []):
        what = f"moves[{i}]" + (f" ({m['move']})" if isinstance(m.get("move"), str) else "")
        nodes = m.get("nodes")
        if not isinstance(nodes, (list, tuple)) or not nodes:
            raise ValueError(f"{what}: binds no nodes")
        for nid in nodes:
            node = program.nodes.get(nid)
            if node is None:
                raise ValueError(f"{what}: no node {nid!r} in the duckscript (its nodes: {', '.join(program.order)})")
            if node.skill not in HOST_SKILLS:
                raise ValueError(f"{what}: node {nid!r} is a {node.skill} node; only walk and stand nodes can run a move")
            if nid in out:
                raise ValueError(f"{what}: node {nid!r} is already bound")
        host = m.get("host") or "walk-stand"
        if host not in ("walk-stand", "raw"):
            raise ValueError(f"{what}: host must be 'walk-stand' or 'raw', got {host!r}")
        cmd = m.get("command")
        if host == "walk-stand" and cmd is not None:
            raise ValueError(f"{what}: a walk-stand move takes no command (only host 'raw' does)")
        if host == "raw":
            if not isinstance(cmd, dict) or set(cmd) != set(nodes):
                raise ValueError(f"{what}: a raw move needs command: {{node id: 13 numbers}} for exactly its nodes")
        ref = m.get("move")
        runner = EXTERNAL if ref == EXTERNAL else (ref if hasattr(ref, "run") else load(ref))
        done = rt.move_done_ticks(m.get("done"))
        for nid in nodes:
            c = None if host != "raw" else command_vector(cmd[nid], m.get("knobs") or ())
            out[nid] = NodeMove(runner, c, done, name=str(ref) if isinstance(ref, str) else type(ref).__name__)
    return out


# ---- the skills layer (skills.ts SkillLayer, for walk / stand / get_up nodes) ----------------------------------------
class SkillLayer:
    """One duck's skill runtime for walk, stand and get_up nodes (and END = stand). Per policy tick: observe the current
    state (it advances the counters of the tick that just ran; not at k = 0), enter a node if the program transitions,
    then step. enter resets done, the skill's tick counter, the get_up rule and the walk heading hold, and keeps the
    motion (#moving: a walk -> walk switch while moving gets no new start kick); after an episodic skill's done the
    duck stands (unless a move is bound). sit / stand_up / kick / roulade lead-ins never trigger on these nodes."""

    def __init__(self, rt: MatchRuntime, stock: dict, moves: dict[str, NodeMove]):
        self.rt, self.stock, self.moves = rt, stock, moves
        self.command = np.zeros(13)
        self.reset()

    def reset(self) -> None:
        self.running = self.skill = "stand"
        self.done = self.done_now = False
        self.node_id = None
        self.params: dict = {}
        self.move: NodeMove | None = None
        self.rule = None
        self.k, self.ran_skill, self.kick, self.psi, self.up, self.tipped, self.moving = 0, False, 0, 0.0, 0, False, False
        self.runner, self.head_in_obs, self.body_in_obs = None, False, False
        self.lead = self.settle = self.still = self.head_reset = self.stand_up_run = 0
        self.head_off = 0.0
        self.fed = np.zeros(13)
        self.fed_now = self.head_now = False
        self.walk_fed = None
        self.yaw_ref = 0.0
        self.head_vals = [math.nan] * 4
        self.head = None  # the tick's held head targets (4, NaN = the move's own), or None

    @property
    def suppress_fallen(self) -> bool:
        """`fallen` is not raised while get_up (or roulade) is still running."""
        return self.skill in ("get_up", "roulade") and not self.done

    def enter(self, node: Node | None) -> None:
        skill = "stand" if node is None else node.skill
        p = {} if node is None else node.params
        self.node_id = None if node is None else node.id
        rt = self.rt
        seated = self.running == "sit" or (self.running == "stand_up" and self.stand_up_run < rt.sit_exit_ticks)
        self.lead = rt.sit_exit_ticks if seated and skill not in ("sit", "stand_up") else 0
        self.settle = rt.settle_max if skill == "roulade" else 0
        self.head_reset = rt.kick_head_max if skill in ("kick_left", "kick_right") else 0
        commanded = skill == "walk" and (p.get("vx") != 0 or p.get("vy") != 0 or p.get("wz") != 0)
        self.kick = KICK_TICKS if commanded and not self.moving else 0
        self.skill, self.params = skill, p
        self.move = None if node is None else self.moves.get(node.id)
        self.rule = self.rt.stock_done[skill] if self.move is None else self.move.done
        self.k, self.ran_skill, self.psi, self.up, self.tipped, self.done = 0, False, 0.0, 0, False, False
        self.fed_now = self.head_now = False
        self.walk_fed = None
        self.yaw_ref = 0.0

    def hold(self, targets: list) -> None:
        """skills.ts hold: this program tick's held head targets (NaN = the move's own)."""
        self.head_vals = list(targets)
        self.head_now = any(math.isfinite(v) for v in targets)

    def feed(self, values: list) -> None:
        """skills.ts feed: a stock walk's vx vy wz (clamped to the walk's ranges), or a raw move's block with its slots."""
        base = None if self.move is None else self.move.command
        if base is None and self.skill == "walk" and values:
            f = self.walk_fed if self.walk_fed is not None else [self.params.get("vx", 0.0), self.params.get("vy", 0.0),
                                                                  self.params.get("wz", 0.0)]
            for i, v in values:
                if i <= 2:
                    lo, hi = self.rt.walk_ranges[i]
                    f[i] = 0.0 if not math.isfinite(v) else min(max(v, lo), hi)
            self.walk_fed = f
            return
        if base is None or not values:
            return
        self.fed[:] = base
        for i, v in values:
            self.fed[i] = v
        self.fed_now = True

    def observe(self, gz: float, gyro_z: float, speed: float = 0.0, head_off: float = 0.0) -> None:
        """gz (projected gravity z), the gyro's z, the horizontal trunk speed (m/s) and the largest head joint offset from
        the default pose (rad) of the current state (after the previous tick)."""
        self.still = self.still + 1 if (speed < self.rt.settle_speed and -gz > self.rt.cos_upright) else 0
        self.head_off = head_off
        was = self.done
        if self.ran_skill:
            self.k += 1
            if self.skill == "walk":
                self.psi += gyro_z * POLICY_DT
            if gz > 0:
                self.tipped = True
            self.up = self.up + 1 if -gz > self.rt.cos_upright else 0
            r = self.rule
            if r is not None and ((r["after"] is not None and self.k >= r["after"])
                                  or (r["upright"] is not None and self.up >= r["upright"])
                                  or (r["rolled"] is not None and self.tipped and self.up >= r["rolled"])):
                self.done = True
        self.done_now = self.done and not was

    def step(self) -> None:
        """The coming policy tick: sets runner (a stock policy's runner, a move's, or EXTERNAL), command, the obs masks."""
        self.done_now = False
        rt = self.rt
        if self.lead > 0:
            self.lead -= 1
            behaviour, self.ran_skill = "stand_up", False
        elif self.settle > 0 and self.still < rt.settle_ticks:
            self.settle -= 1
            behaviour, self.ran_skill = "stand", False
        elif self.head_reset > 0 and self.head_off > rt.kick_head_tol:
            self.head_reset -= 1
            behaviour, self.ran_skill = "stand", False
        elif self.done and rt.kind[self.skill] == "episodic" and self.move is None:
            behaviour, self.ran_skill = "stand", False
        else:
            behaviour, self.ran_skill = self.skill, True
        if self.ran_skill:
            self.settle = self.head_reset = 0
        move = self.move if self.ran_skill else None
        c = self.command
        if move is not None and move.command is not None:
            c[:] = self.fed if self.fed_now else move.command  # the raw host: the move's own block, head and body in the obs
            self.head_in_obs = self.body_in_obs = True
            self.moving = c[0] != 0 or c[1] != 0 or c[2] != 0
        else:
            c[:] = 0.0
            p = self.params
            c[4], c[5], c[6] = p.get("head_pitch", 0.0), p.get("head_yaw", 0.0), p.get("head_roll", 0.0)
            moving = False
            if behaviour == "walk":
                fed = self.walk_fed
                if fed is not None:
                    vx, vy, wz = fed
                    self.yaw_ref += wz * POLICY_DT
                else:
                    vx, vy, wz = p.get("vx", 0.0), p.get("vy", 0.0), p.get("wz", 0.0)
                moving = vx != 0 or vy != 0 or wz != 0
                if moving:
                    k = self.k
                    if fed is not None and not self.moving and vx != 0 and k >= self.kick:
                        self.kick = k + KICK_TICKS  # a feed that starts a standing walk gets the start kick
                    c[0] = (KICK_COMMAND if vx >= 0 else -KICK_COMMAND) if k < self.kick else walk_command_vx(vx)
                    c[1] = _clip(vy * LATERAL_GAIN, LATERAL_MAX)
                    ref = self.yaw_ref if fed is not None else wz * k * POLICY_DT
                    c[2] = _clip(wz + YAW_GAIN * (ref - self.psi), WZ_LIMIT)
            elif behaviour == "sit":
                c[0] = 1.0
            elif behaviour == "ground_pick":
                phi = self.rt.ground_pick_phase * self.k
                c[0] = math.cos(2 * math.pi * phi)
                c[1] = math.sin(2 * math.pi * phi)
            self.moving = moving
            runner, self.head_in_obs, self.body_in_obs = self.stock[self.rt.policy[behaviour]]
            self.runner = runner
        if move is not None:
            self.runner = move.runner
        self.head = self.head_vals if (self.ran_skill and self.head_now) else None
        self.stand_up_run = self.stand_up_run + 1 if behaviour == "stand_up" else 0
        self.running = behaviour


# ---- sensors (runtime/sensors.ts, match mode) ------------------------------------------------------------------------
class Sensors:
    """Program-visible signals of one duck at program ticks, from the forwarded state (sensors.ts Sensors with the match
    options: the duck's own trunk, gyro, free joint and sites; target_* = the opponent's head body; height from the
    stage top). Rays only for the signals the program reads. `duck` needs root_qpos_adr, trunk, gyro_adr, camera_site,
    ray_site (duel.py DuckIndex)."""

    def __init__(self, model: mujoco.MjModel, rt: MatchRuntime, program: CompiledProgram, prng, duck, target_body: int,
                 goal: dict | None, height_base: float, rival: tuple[int, int] | None = None, crumbs=None):
        used = program.signals
        self.crumbs = crumbs  # Bread Fight: crumbs.py CrumbSense of this duck's slot (None: no crumb field)
        self.see_crumb = any(s in used for s in CRUMB_SIGNALS)
        self.rival = rival  # (the other duck's trunk body, this slot's role)
        self.see_rival = any(s in used for s in RIVAL_SIGNALS)
        self.see_hill = any(s in used for s in HILL_SIGNALS)
        self.log_y = rt.log[1] if rt.log is not None and "log_off" in used else None
        self.log_dof = -1
        if rt.log is not None and "log_spin" in used:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{rt.log[0]}_hinge")
            if j < 0:
                raise ValueError(f"log_spin: no joint {rt.log[0]}_hinge in the world")
            self.log_dof = int(model.jnt_dofadr[j])
        self.m, self.rt, self.prng, self.duck = model, rt, prng, duck
        if isinstance(target_body, dict):  # a solo Challenge's fixed target point (episode.ts)
            self.target = ((float(target_body["x"]), float(target_body["y"]), float(target_body["z"])), -1)
            target_body = -1
        else:
            self.target = None if int(target_body) < 0 else (None, int(target_body))
        self.goal = None if goal is None else ((float(goal["x"]), float(goal["y"]), float(goal["z"])), -1)
        self.height_base = float(height_base)
        self.ray = "obstacle_dist" in used
        self.see_target = any(s in used for s in TARGET_SIGNALS)
        self.see_goal = any(s in used for s in GOAL_SIGNALS)
        self.progress = "progress" in used
        self.prev_x = self.prev_y = math.nan
        self.touch_geoms = None
        if any(x in used for x in TOUCH_SIGNALS) and target_body > 0:  # sensors.ts: 1 = own geom, 2 = the target's
            root = model.body_rootid
            mine, theirs = root[duck.trunk], root[target_body]
            gb = root[model.geom_bodyid]
            self.touch_geoms = np.where(gb == mine, 1, np.where((gb == theirs) & (theirs != 0), 2, 0)).astype(np.int64)
        self.values: dict = {}
        self._geomid = np.zeros(1, np.int32)
        self._pnt, self._vec = np.zeros(3), np.zeros(3)

    def compute(self, d: mujoco.MjData, n: int, n_enter: int, fallen: bool, done: bool) -> dict:
        v, duck, rt = self.values, self.duck, self.rt
        qpos, xm = d.qpos, d.xmat[duck.trunk]
        a = duck.root_qpos_adr
        tilt = math.acos(max(-1.0, min(1.0, xm[8]))) * 180 / math.pi
        x, y = qpos[a], qpos[a + 1]
        v["t"] = n / rt.program_hz
        v["t_state"] = (n - n_enter) / rt.program_hz
        v["tilt"] = tilt
        v["upright"] = tilt < rt.upright_tilt_deg
        v["fallen"] = fallen
        v["done"] = done
        v["speed"] = 0.0 if math.isnan(self.prev_x) else js_hypot2(x - self.prev_x, y - self.prev_y) / (1 / rt.program_hz)
        dxv = 0.0 if math.isnan(self.prev_x) else (x - self.prev_x) / 0.1
        dyv = 0.0 if math.isnan(self.prev_y) else (y - self.prev_y) / 0.1
        self.prev_x, self.prev_y = x, y
        v["yaw_rate"] = d.sensordata[duck.gyro_adr + 2]
        v["height"] = qpos[a + 2] - self.height_base
        if self.progress:
            v["progress"] = 0.0  # runMatch feeds progress 0 (a duel Challenge has no metric so far)
        if self.ray:
            v["obstacle_dist"] = self._obstacle(d)
        heading = math.atan2(xm[3], xm[0])
        v["heading"] = heading
        v["forward_speed"] = dxv * math.cos(heading) + dyv * math.sin(heading)
        v["side_speed"] = dyv * math.cos(heading) - dxv * math.sin(heading)
        if self.see_target:
            v["target_seen"], v["target_bearing"], v["target_dist"] = self._look(d, self.target, heading)
        if self.touch_geoms is not None:
            v["touching"], v["touch_bearing"] = self._touch(d, x, y, heading)
        if self.see_goal:
            if rt.goal_around and self.goal is not None and self.goal[0] is not None:  # chase.around: no cone, no noise
                gx, gy, _ = self.goal[0]
                v["goal_seen"], v["goal_bearing"] = True, wrap_pi(math.atan2(gy - y, gx - x) - heading)
                v["goal_dist"] = js_hypot2(gx - x, gy - y)
            else:
                v["goal_seen"], v["goal_bearing"], v["goal_dist"] = self._look(d, self.goal, heading)
        if self.see_rival:
            if self.rival is None:
                v["rival_bearing"], v["rival_dist"], v["role"] = 0.0, math.inf, 0
            else:
                body, role = self.rival
                dx, dy = d.xpos[body, 0] - x, d.xpos[body, 1] - y
                v["rival_bearing"] = wrap_pi(math.atan2(dy, dx) - heading)
                v["rival_dist"], v["role"] = js_hypot2(dx, dy), role
        if self.see_hill:
            if rt.hill_pad is None:
                v["hill_bearing"], v["hill_dist"] = 0.0, 5.0
            else:
                px, py = rt.hill_pad
                v["hill_bearing"] = wrap_pi(math.atan2(py - y, px - x) - heading)
                v["hill_dist"] = min(5.0, js_hypot2(px - x, py - y))
        if self.see_crumb:  # sensors.ts: the nearest crumb, then the second-nearest, then the counts
            cr = self.crumbs
            for pre, c in (("crumb", None if cr is None else cr.at(x, y)), ("crumb2", None if cr is None else cr.second(x, y))):
                if c is None:
                    v[pre + "_bearing"], v[pre + "_dist"], v[pre + "_rival_dist"] = 0.0, 5.0, 5.0
                else:
                    v[pre + "_bearing"] = wrap_pi(math.atan2(c[1] - y, c[0] - x) - heading)
                    v[pre + "_dist"] = js_hypot2(c[0] - x, c[1] - y)
                    v[pre + "_rival_dist"] = cr.rival_dist_to(c)
            v["crumbs"] = 0 if cr is None else cr.mine()
            v["rival_crumbs"] = 0 if cr is None else cr.rivals()
        if self.log_y is not None:  # + = toward +y of the log's top line
            v["log_off"] = js_fixed(max(-2.0, min(2.0, y - self.log_y)))
        if self.log_dof >= 0:  # + = the log's top moves toward +y
            v["log_spin"] = js_fixed(-d.qvel[self.log_dof])
        return v

    def _touch(self, d, x: float, y: float, heading: float):
        """sensors.ts #touch: (touching, the bearing of the contacts' mean point) for contacts (dist <= 0) between the
        duck's body and the target's."""
        g = self.touch_geoms
        n, sx, sy = 0, 0.0, 0.0
        for c in range(d.ncon):
            con = d.contact[c]
            a, b = g[con.geom1], g[con.geom2]
            if a + b == 3 and a * b == 2 and con.dist <= 0:
                sx += con.pos[0]
                sy += con.pos[1]
                n += 1
        if n == 0:
            return False, 0.0
        return True, V2.wrap_angle(math.atan2(sy / n - y, sx / n - x) - heading)

    def _cast(self, d, frm, dx, dy, dz, exclude: int) -> float:
        p, u = self._pnt, self._vec
        p[0], p[1], p[2] = frm[0], frm[1], frm[2]
        u[0], u[1], u[2] = dx, dy, dz
        return mujoco.mj_ray(self.m, d, p, u, self.rt.groups, True, exclude, self._geomid)

    def _obstacle(self, d) -> float:
        i = self.duck.ray_site
        m = d.site_xmat[i]
        r = self._cast(d, d.site_xpos[i], m[0], m[3], m[6], -1)
        return r if 0 <= r <= self.rt.ray_max else math.inf

    def _look(self, d, point, heading: float):
        """(seen, bearing, dist) of a point from the head camera; draws the two noise values always (sensors.ts #look)."""
        if point is None:
            return False, 0.0, math.inf
        rt = self.rt
        nb = self.prng.uniform(-rt.noise_bearing, rt.noise_bearing)
        nd = self.prng.uniform(-rt.noise_dist, rt.noise_dist)
        fixed, body = point
        if fixed is not None:
            px, py, pz = fixed
        else:
            px, py, pz = d.xpos[body]
        i = self.duck.camera_site
        c, m = d.site_xpos[i], d.site_xmat[i]
        dx, dy, dz = px - c[0], py - c[1], pz - c[2]
        horiz = js_hypot2(dx, dy)
        d3 = js_hypot2(horiz, dz)
        cos_angle = (dx * m[0] + dy * m[3] + dz * m[6]) / d3 if d3 > 0 else 1.0
        if horiz > rt.view_max_dist or cos_angle < rt.cos_view:
            return False, 0.0, math.inf
        if d3 > rt.los_tol:
            hit = self._cast(d, c, dx / d3, dy / d3, dz / d3, body)
            if 0 <= hit < d3 - rt.los_tol:
                return False, 0.0, math.inf
        bearing = math.atan2(dy, dx) - heading
        bearing -= 2 * math.pi * js_round(bearing / (2 * math.pi))
        return True, bearing + nb, horiz * (1 + nd)


SOLO_SIGNALS = ("maze_yaw",)  # sensors.ts signals a solo program may read beyond KNOWN_SIGNALS (no mover needed)


class SoloSensors(Sensors):
    """sensors.ts Sensors in solo mode (episode.ts runEpisode; kit 1.9.0, env.py program mode): no opponent; target_* /
    goal_* = the Challenge's points (challenge.target / challenge.goal: a fixed {x, y, z} or a {prop} body, as sensors.ts resolvePoint;
    none: unseen, False, 0, inf, no noise drawn, as #look(null)), seen from the head camera with the noise drawn from
    `prng` (the caller passes the episode stream past drawRandomisation, as episode.ts does); height from the floor;
    progress = `progress_value` (the caller sets the row's metric before the program tick) when the Challenge allows the
    signal, else 0; maze_yaw = the heading off the nearest quarter turn; the Challenge's movers' signals
    (movers.Mechanics.signals at t = n / 10 and the trunk's x, y). Values a TS sensor rounds (+x.toFixed(6)) are
    rounded so here too (movers.fixed6)."""

    def __init__(self, model: mujoco.MjModel, rt: MatchRuntime, program: CompiledProgram, prng, duck, mech=None,
                 allows_progress: bool = False, target: dict | None = None, goal: dict | None = None):
        super().__init__(model, rt, program, prng, duck, -1, None, 0.0)
        self.target, self.goal = self._point(model, target), self._point(model, goal)
        used = program.signals
        self.mech = mech
        self.see_maze_yaw = "maze_yaw" in used
        self.see_mech = mech is not None and any(s in used for s in mech.signal_names)
        self.allows_progress = bool(allows_progress)
        self.progress_value = 0.0

    @staticmethod
    def _point(model: mujoco.MjModel, p: dict | None):
        """sensors.ts resolvePoint: ((x, y, z), -1) for a fixed point, (None, body) for {prop}, None for none."""
        if not p:
            return None
        if "prop" in p:
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p["prop"])
            if b < 0:
                raise ValueError(f"solo sensors: no body {p['prop']!r} for a challenge point")
            return None, int(b)
        return (float(p["x"]), float(p["y"]), float(p["z"])), -1

    def compute(self, d: mujoco.MjData, n: int, n_enter: int, fallen: bool, done: bool) -> dict:
        from .movers import fixed6
        v = super().compute(d, n, n_enter, fallen, done)
        if self.progress:
            v["progress"] = self.progress_value if self.allows_progress else 0.0
        if self.see_maze_yaw:
            xm = d.xmat[self.duck.trunk]
            heading = math.atan2(xm[3], xm[0])
            v["maze_yaw"] = fixed6(heading - (math.pi / 2) * js_round(heading / (math.pi / 2)))
        if self.see_mech:
            a = self.duck.root_qpos_adr
            self.mech.signals(v["t"], float(d.qpos[a]), float(d.qpos[a + 1]), v)
        return v


class FallEvents:
    """episode.ts FallEvents: one `fall` when `fallen` rises, then nothing until upright again (`recovered`)."""

    def __init__(self):
        self.down = False

    def update(self, fallen: bool, upright: bool) -> str | None:
        if not self.down and fallen:
            self.down = True
            return "fall"
        if self.down and upright:
            self.down = False
            return "recovered"
        return None


# ---- one duck's program runtime (match.ts MatchDuck, without the judge's and the flags' parts) ------------------------
class DuckProgram:
    """One duck's duckscript runtime in a match: CompiledProgram + its bound moves, run on a duck slot per episode.
    reset(...) binds it to a slot (its index set, its PRNG stream, the opponent's head body); then per policy tick k:
    observe_row(cos_tilt) on the row's state, step(k, d) (d forwarded at row k). After step: runner (a stock policy's
    or a move's runner, or EXTERNAL), command (13), head_in_obs, body_in_obs, external, node (Node or None = END),
    node_index (frame node index, -1 = END), skill_index (the running behaviour in schema SKILL_NAMES order).
    events: node_enter / done / fall / recovered as the TS episode records them (t, kind, node, via, rule)."""

    def __init__(self, rt: MatchRuntime, program: CompiledProgram, stock: dict, moves: dict[str, NodeMove] | None = None,
                 routes: dict | None = None):
        self.rt, self.program = rt, program
        self.moves = moves or {}
        self.state = V2.ProgramState(program) if program.v2 else None  # duck-program/2: memory, vars, BACK
        if self.state is not None and routes is not None:
            self.state.world = V2.RouteWorld(routes)
        elif self.state is not None and getattr(program, "uses_routes", False):
            raise ValueError(f"{program.name}: uses path()/path_len() but the Challenge gives no routes (info has no routes)")
        self.layer = SkillLayer(rt, stock, self.moves)
        self.external_nodes = {nid for nid, mv in self.moves.items() if mv.runner == EXTERNAL}

    def reset(self, model: mujoco.MjModel, duck, prng, target_body: int, goal: dict | None, height_base: float,
              rival: tuple[int, int] | None = None, crumbs=None, sensors: Sensors | None = None) -> None:
        """sensors: a solo episode's (SoloSensors); else the match's Sensors from the other arguments (rival: the other
        duck's trunk body and this slot's role, kit 1.9.0; crumbs: Bread Fight's CrumbSense of this slot, kit 1.10.0)."""
        self.duck = duck
        self.sensors = sensors if sensors is not None else Sensors(model, self.rt, self.program, prng, duck, target_body,
                                                                  goal, height_base, rival, crumbs)
        self.falls = FallEvents()
        self.layer.reset()
        self.node: Node | None = None
        self.node_name = self.program.start
        self.node_index, self.n_enter = 0, 0
        self.fall_count, self.upright, self.cos_tilt = 0, True, 1.0
        self.events: list[tuple] = []
        self.signals: dict | None = None
        self.runner, self.external = None, False
        self.x = self.y = 0.0
        if self.state is not None:
            self.state.reset()

    @property
    def command(self) -> np.ndarray:
        return self.layer.command

    @property
    def head_in_obs(self) -> bool:
        return self.layer.head_in_obs

    @property
    def body_in_obs(self) -> bool:
        return self.layer.body_in_obs

    @property
    def head(self):
        """The tick's held head targets (4 values, NaN = the move's own), or None (duck-program/2 head)."""
        return self.layer.head

    @property
    def skill_index(self) -> int:
        return self.rt.skill_index[self.layer.running]

    def observe_row(self, cos_tilt: float) -> None:
        """match.ts observeRow's posture part: upright (tilt < 32 deg) and the `fallen` counter (tilt > 60 deg rows)."""
        self.cos_tilt = cos_tilt
        tilt = math.acos(max(-1.0, min(1.0, cos_tilt))) * 180 / math.pi
        self.upright = tilt < self.rt.upright_tilt_deg
        self.fall_count = self.fall_count + 1 if tilt > self.rt.fallen_tilt_deg else 0

    def _enter(self, name: str, n: int, why, s: dict | None = None) -> None:
        node = None if name == self.rt.end else self.program.nodes[name]
        self.node_name, self.node = name, node
        self.node_index = -1 if node is None else node.index
        self.n_enter = n
        if self.state is not None:
            self.state.enter(node, s)
        self.layer.enter(node)
        via, rule = ("start", None) if why is None else why[1:]
        self.events.append((n / self.rt.program_hz, "node_enter", name, via, rule))

    def step(self, k: int, d: mujoco.MjData) -> None:
        """match.ts MatchDuck.step: the skills layer's counters, the program tick (every PROGRAM_EVERY rows), the step."""
        rt, layer = self.rt, self.layer
        if k == 0:
            self._enter(self.program.start, 0, None)
        else:
            dk = self.duck
            v = d.qvel[dk.root_dof_adr:dk.root_dof_adr + 2]
            q = d.qpos
            off = 0.0
            for j in rt.head_joints:  # skills.ts headOffset
                off = max(off, abs(q[dk.qadr[j]] - dk.default_pos[j]))
            layer.observe(-self.cos_tilt, d.sensordata[dk.gyro_adr + 2], js_hypot2(float(v[0]), float(v[1])), off)
        if layer.done_now:
            self.events.append((k / rt.ticks_per_s, "done", self.node_name, None, None))
        if k % rt.program_every == 0:
            n = k // rt.program_every
            fallen = self.fall_count >= rt.fallen_ticks and not layer.suppress_fallen
            s = self.sensors.compute(d, n, self.n_enter, fallen, layer.done)
            self.signals = s
            fall = self.falls.update(fallen, self.upright)
            if fall is not None:
                self.events.append((n / rt.program_hz, fall, self.node_name, None, None))
            if self.state is not None:  # duck-program/2: ProgramRunner.tick (memory every tick, any rules, BACK)
                w = self.state.world
                if w is not None:
                    w.x, w.y, w.heading = self.x, self.y, s.get("heading", 0.0)
                go = self.state.tick(self.node, s)
                if go is not None:
                    self._enter(go[0], n, go, s)
                node = self.node
                if node is not None and node.feed:
                    self.layer.feed(self.state.feed_values(node, s))
                if node is not None and node.head:
                    self.layer.hold(self.state.head_targets(node, s))
            else:
                go = None if self.node is None else CompiledProgram.next_transition(self.node, s)
                if go is not None:
                    self._enter(go[0], n, go)
        layer.step()
        self.runner = layer.runner
        self.external = self.runner == EXTERNAL
