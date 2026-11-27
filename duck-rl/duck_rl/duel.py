"""The two-duck world of a duel Challenge (PvP: Sumo) in Python: packages/sim/src/runtime/match.ts runMatch, row for row, so
a training env can play the learner's duck against a frozen opponent and a match replays a TS one.

  world      ts/prepare.ts's export of the Challenge (buildMatchWorldXml: duck 0 with the plant's names, duck 1 prefixed
             "b_"; the info's `duel` block: rules, both slots' indices, runtime constants), compiled once per process
             (physics only, as the sim compiles it); each DuelWorld owns a model copy (the draws and the BAM write into
             it) and one MjData holding both ducks, each with its own index set and NumbaBam (lab.duck);
  sides      A (side 0) and B (side 1): a duckscript or a duck-strategy/1 each (program.py DuckProgram; a node's
             move may be EXTERNAL: the caller supplies that side's action on the ticks the node runs). reset(label,
             swapped) puts A in slot 0 (swapped: slot 1), B in the other: slot i spawns at duel.spawns[i] and draws
             from stream i, as the TS match swaps sides on one seed;
  reset      match.ts:323-357 with World's resets: each slot's draws from its own stream (prng.py; dx, dy, dyaw, sole
             friction, trunk mass), the spawn-line rotation from the ":match" stream, friction / mass / inertia into
             the model, mj_setConst, the STAND keyframe for the whole data, both servo poses and ctrl, both spawn
             poses (rotated about the ring centre, jittered, raised by spawn z), loop state and BAM state per duck;
             `label` is the PRNG label base: "sumo:3:<seed>" replays public seed <seed> of the TS match (prng.py
             episode_label; a hidden seed's "sumo:3:key:<hex>" too), any other string is a training seed;
  per row k  match.ts:418-485: mj_forward -> non-finite root qpos (before the decision: invalid; after: the episode
             stops) or, after the decision, MuJoCo instability (stop) -> each slot's posture rows -> the judge
             (metrics.py MatchPoints) -> stop at the decision's tail end -> both programs' ticks (slot 0, then slot 1)
             -> both obs (lab.fastenv.nb_obs: head EMA, lagged joint velocities, prev action, the tick's command)
             -> step(actions): each slot's policy (SimdMlp, bit-identical to the TS runner) or the caller's action
             for an EXTERNAL tick -> targets -> both BAMs latch -> decimation x (BAM slot 0, BAM slot 1, mj_step).
DuelRow (one per row, reused: copy what you keep) is indexed by SIDE (0 = A, 1 = B; row.slot[side] = its slot): obs
(61), priv (the 6 critic extras of lab.fastenv's nb_obs with the height from the stage top: heading-frame trunk
velocity x2, 2 vz, 10 (z - top - 0.12), friction - 1, 10 (mass - 1)), external, node, skill, dist / z / tilt /
upright / down / down_rows, the ring radius, touching (duck-duck contact this row), penetration, and the decision.

Ported rules (kit 1.9.0, LB-108): Sumo's (out, down, the shrinking ring, the time-out), win "goose_chase" (the
duck-duck touch per row feeds the judge), timeout "both_lose", Wobbly Raft's swell (a raft prop with swell_nm > 0:
match.ts applySwell's torque in qfrc_applied of the raft's ball joint, its phase and sense drawn from the ":match"
stream after the spawn rotation; Log Roll v2, LB-109: the same on a log's hinge, a steady push when swell_period_s
is 0), DuelRules.low_out (Wobbly Raft v3: a trunk low on the stage top too long is out, "too_low"; metrics.py MatchPoints with ts/prepare.ts's runtime low_ref) and the signals of program.py. A Challenge with anything else the TS match runs (more than
two ducks, another win rule, a driven prop, a rules block not listed in PORTED_RULES) is refused (ValueError), so
training never optimises a different game.
Kit 1.10.0: win "bread" (Bread Fight v2 1v1, DuelRules.bread without flee / cells / goose / brood): the crumb field
(duck_rl/crumbs.py) is made in the reset from the ":match" stream after the spawn rotation (match.ts creates it there;
it draws nothing until a crumb is placed), updated by the judge on every row until the decision with both heads' xy,
and its props' free joints (`<crumb id>_free`) are written once per tick before the substeps (prop-drive.ts: the poses
for time (k + 1) / ticks_per_s, qvel zeroed; at the reset for t = 0); the crumb_* / crumb2_* / crumbs / rival_crumbs
signals read it (program.py). start_hook(world) (the raw host in duels, from the 2026-10-15 night): called in the
reset after the draws and starts, before the poses (duel_env.py draws a raw command there); None = nothing.

Parity with TS (tools/duel_parity.py): the draws, starts, MLP actions (simd_mlp.py), programs, sensor noise and judge
are exact (on the TS trajectory's states the Python programs and judge reproduce every node, event and decision), but
the physics is not: native MuJoCo (the pip wheel) and the sim's WASM build of the same version compute a different
multi-contact set for the left sole hull on the stage's flat top (the right one agrees), from the first touchdown
(row 2) on, with the compiled meshes identical and the set stable under 3e-8 perturbations in Python; the episodes
then diverge chaotically (> 1 mm by row ~15). Outcomes agree in distribution (house-medium vs house-easy, 32 seeds x 2
sides: mean points of A 0.719 Python vs 0.688 TS), not episode by episode. Training workers: compile the duckscripts
once (compile_program, one node process each) and pass the compiled dicts; ~0.33 CPU-s per 20-s episode, mostly mj_step.

    from duck_rl.duel import DuelWorld
    w = DuelWorld("packages/challenges/baselines/sumo/house-medium.json", "packages/challenges/baselines/sumo/house-easy.json")
    row = w.reset("sumo:3:0")             # TS public seed 0, A in slot 0
    while not row.stopped: row = w.step()  # all internal; w.result() -> winner side, reason, t, points
    # a learner on side A: w.set_side(0, "my.json", moves=[{"nodes": ["push"], "move": "external"}]); then
    # row = w.reset(label, swapped); row = w.advance(0) -> a row where A acts (row.external[0]) or the end;
    # row = w.step({0: action}) ...
"""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numba import njit

from . import PLANT, bridge, ensure_scene, load_info
from .crumbs import CrumbField, CrumbSense, crumb_ids
from .metrics import MatchPoints
from .program import EXTERNAL, CompiledProgram, DuckProgram, MatchRuntime, bind_moves
from .prng import draw_randomisation, match_prng
from .simd_mlp import SimdMlp
from duckref.bam import BamConfig
from duckref.plant import load_manifest
from lab.duck import STAND_Z, NumbaBam
from lab.fastenv import nb_obs

NUM_JOINTS = 14
WARNINGS = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL)
_PROGRAMS: dict = {}
_RUNNERS: dict[str, SimdMlp] = {}
# DuelRules keys the Python match runs (schema challenge.ts DuelRules); any other key is refused (check_ported)
PORTED_RULES = ("spawns", "rotate_spawns", "ring", "platform", "out", "down_loses_s", "timeout", "draw_band_m",
                "tail_s", "match_seeds", "win", "chase", "low_out", "bread", "finish")  # finish: the race duels (kit 1.15.0)
PORTED_WINS = (None, "goose_chase", "bread")
BREAD_REFUSED = ("flee", "cells", "goose", "brood")  # crumbs.py REFUSED: Laser Dot, Duck-Man, Mama's Brood
# scene props the TS match drives or senses with its own runtime (match.ts: sweep, wall sweep, cannons, gusts,
# roomba); a raft and a log are ported (their swell), passive props need nothing
DRIVEN_PROPS = ("sweeper", "wall_sweep", "cannons", "gust_pier", "roomba")


def check_ported(info: dict) -> None:
    """ValueError when the duel Challenge of `info` needs a part of match.ts the Python match does not run (module doc)."""
    duel, challenge = info["duel"], info["challenge"]["id"]
    r = duel["rules"]
    why = []
    if len(duel["ducks"]) != 2 or len(r["spawns"]) != 2:
        why.append(f"{max(len(duel['ducks']), len(r['spawns']))} ducks (the Python match is 1v1)")
    extra = sorted(k for k in r if k not in PORTED_RULES and r[k] not in (None, False))
    if extra:
        why.append(f"rules {', '.join(extra)}")
    if r.get("win") not in PORTED_WINS:
        why.append(f"win {r.get('win')!r}")
    if r.get("timeout") not in ("draw", "closer_to_centre", "both_lose"):  # schema DUEL_TIMEOUTS
        why.append(f"timeout {r.get('timeout')!r}")
    driven = sorted({p["kind"] for p in duel.get("scene") or [] if p["kind"] in DRIVEN_PROPS})
    if driven:
        why.append(f"driven props {', '.join(driven)}")
    bread = r.get("bread") or {}
    variants = [k for k in BREAD_REFUSED if bread.get(k) is not None]
    if variants:
        why.append(f"bread {', '.join(variants)}")
    if why:
        raise ValueError(f"duel challenge {challenge}: the Python match runtime does not run {'; '.join(why)} (ported: Sumo's "
                         "rules, win goose_chase, timeout both_lose, the raft / log swell, low_out, Bread Fight 1v1's "
                         "crumbs, the race finish)")


@njit(cache=True)
def _contacts(cgeom, ncon, owner, efc_type, efc_pos, nefc):
    """(the ducks touch: a contact between a geom of slot 0 and one of slot 1, the deepest contact penetration (m) as
    match.ts measures it: max -efc_pos over the contact rows, type >= mjCNSTR_CONTACT_FRICTIONLESS)."""
    touch = False
    for c in range(ncon):
        a, b = owner[cgeom[c, 0]], owner[cgeom[c, 1]]
        if a >= 0 and b >= 0 and a != b:
            touch = True
            break
    pen = 0.0
    for i in range(nefc):
        if efc_type[i] >= 5 and -efc_pos[i] > pen:
            pen = -efc_pos[i]
    return touch, pen


@dataclass
class DuckIndex:
    """One duck slot's model indices (plant.ts DuckIndex + match.ts MatchSideModel), resolved by name with its prefix
    and checked against the TS export (info duel.ducks[slot])."""
    slot: int
    prefix: str
    root_qpos_adr: int
    root_dof_adr: int
    qadr: np.ndarray  # servo joints' qpos addresses, ONNX joint order
    vadr: np.ndarray
    act: np.ndarray  # the actuator (ctrl index) driving each servo joint
    gyro_adr: int
    trunk: int
    head_body: int  # the body carrying its head_camera site: the opponent's target_* point
    camera_site: int
    ray_site: int
    soles: list
    base_mass: float
    base_inertia: np.ndarray
    default_pos: np.ndarray


def resolve_duck(m: mujoco.MjModel, manifest: dict, slot: int, want: dict, rt_raw: dict) -> DuckIndex:
    """plant.ts resolveDuck for the duck whose names carry `want["prefix"]`, plus its head, sites and soles; ValueError
    if anything differs from the TS export `want` (another MJCF compile, a stale scene)."""
    p = want["prefix"]
    jids = [m.joint(p + n).id for n in manifest["joint_names"]]
    acts = []
    for j in jids:
        a = [i for i in range(m.nu) if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[i, 0] == j]
        if len(a) != 1:
            raise ValueError(f"joint {p}{m.joint(j).name} is driven by {len(a)} actuators, expected 1")
        acts.append(a[0])
    trunk = m.body(p + "trunk_base").id
    root = int(m.body_jntadr[trunk])
    if root < 0 or m.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"{p}trunk_base has no free joint first")
    cam = m.site(p + rt_raw["camera_site"]).id
    d = DuckIndex(slot=slot, prefix=p, root_qpos_adr=int(m.jnt_qposadr[root]), root_dof_adr=int(m.jnt_dofadr[root]),
                  qadr=np.array([m.jnt_qposadr[j] for j in jids], np.int64),
                  vadr=np.array([m.jnt_dofadr[j] for j in jids], np.int64), act=np.array(acts, np.int64),
                  gyro_adr=int(m.sensor_adr[m.sensor(p + "imu_ang_vel").id]), trunk=trunk,
                  head_body=int(m.site_bodyid[cam]), camera_site=cam, ray_site=m.site(p + rt_raw["ray_site"]).id,
                  soles=[m.geom(p + n).id for n in rt_raw["sole_geoms"]], base_mass=float(m.body_mass[trunk]),
                  base_inertia=m.body_inertia[trunk].copy(), default_pos=np.array(manifest["default_pos"], np.float64))
    mine = dict(prefix=p, root_qpos_adr=d.root_qpos_adr, root_dof_adr=d.root_dof_adr, servo_qpos_adr=d.qadr.tolist(),
                servo_dof_adr=d.vadr.tolist(), servo_ctrl=d.act.tolist(), gyro_adr=d.gyro_adr, trunk_body=d.trunk,
                head_body=d.head_body, camera_site=d.camera_site, ray_site=d.ray_site, soles=d.soles,
                base_mass=d.base_mass, base_inertia=d.base_inertia.tolist())
    diff = {k: (mine[k], want[k]) for k in mine if mine[k] != want[k]}
    if diff:
        raise ValueError(f"duck slot {slot}: the Python model differs from the TS export (re-export the scene: "
                         f"duck_rl.ensure_scene(challenge, force=True)): {diff}")
    return d


def world_model(scene: Path) -> mujoco.MjModel:
    """A fresh copy of the compiled two-duck world (env.py world_model: physics only, parsed once per process)."""
    from .env import world_model as wm
    return wm(str(scene))


def compile_program(challenge: str, program, base: Path | None = None) -> dict:
    """A duckscript (a path, or the JSON object) through the TS validator and compiler (ts/prepare.ts --program-full):
    the compiled program as the sim runs it. Paths are cached per process by (challenge, path, mtime)."""
    with tempfile.TemporaryDirectory() as td:
        if isinstance(program, (str, Path)):
            path = Path(program).expanduser()
            if not path.is_absolute():
                path = base / path if base is not None and (base / path).exists() else _resolve(path)
            key = (challenge, str(path.resolve()), path.stat().st_mtime)
            if key in _PROGRAMS:
                return _PROGRAMS[key]
        else:
            key, path = None, Path(td) / "program.json"
            path.write_text(json.dumps(program))
        out = Path(td) / "full.json"
        r = bridge.run_ts("prepare", ["--challenge", challenge, "--program", path.resolve(), "--program-full", out],
                          capture_output=True, text=True)
        if r.returncode != 0:
            raise ValueError(f"prepare --program {path} failed (exit {r.returncode}): an invalid duckscript?\n{r.stdout}\n{r.stderr}")
        full = json.loads(out.read_text())
    if key is not None:
        _PROGRAMS[key] = full
    return full


def _resolve(p: Path) -> Path:
    """A relative path: from the working directory, else the kit, else the repo (the stock house bots)."""
    from . import KIT, REPO
    for b in (Path.cwd(), KIT, REPO):
        if b is not None and (b / p).exists():
            return (b / p).resolve()
    raise FileNotFoundError(f"{p}: not found from {Path.cwd()}, {KIT} or {REPO}")


def load_runner(ref, base: Path | None = None) -> SimdMlp:
    """A move ref -> its runner (cached per process): an x.mlp.json path (relative to `base`, e.g. the strategy file's
    folder), a move id mv_<12 hex> or an alias <slug>@<n> (the kit's move library, bridge.move_path)."""
    from .moves import is_library_ref
    s = str(ref)
    if is_library_ref(s):
        path = Path(bridge.move_path(s)["json"])
    else:
        path = Path(s).expanduser()
        if not path.is_absolute():
            path = (base / path) if base is not None and (base / path).exists() else _resolve(path)
        from .moves import bin_of, resolve_path
        if path.exists() and not bin_of(path).exists():  # the player package's .bin: the CLI's copy (fetched on first use)
            path = Path(resolve_path(path)["json"])
    path = path.resolve()
    if str(path) not in _RUNNERS:
        _RUNNERS[str(path)] = SimdMlp(path)
    return _RUNNERS[str(path)]


class DuelSide:
    """One side of a match: a compiled duckscript and its node moves. `spec`: a duckscript (path or JSON object), a
    duck-strategy/1 (path or object: its program and moves, paths relative to its file), a compiled program (dict from
    --program-full, or a CompiledProgram). `moves` (duck-strategy/1 entries, added to a strategy's own): {nodes, move,
    host?, command?, done?, pin?}; move = an x.mlp.json path, a move id / alias, a loaded runner, or EXTERNAL."""

    def __init__(self, challenge: str, rt: MatchRuntime, spec, moves: list[dict] | None = None, name: str | None = None):
        base, entries = None, list(moves or [])
        if isinstance(spec, (str, Path)):
            path = Path(spec).expanduser()
            path = path if path.is_absolute() else _resolve(path)
            base, name = path.parent, name or path.stem
            spec = json.loads(path.read_text())
            if spec.get("format") in ("duck-program/1", "duck-program/2") and not _is_compiled(spec):
                spec = compile_program(challenge, path)
        if isinstance(spec, dict) and spec.get("format") in ("duck-strategy/1", "duck-rl-submission/1"):
            entries = [dict(e, _base=base) for e in spec.get("moves", [])] + entries
            spec = spec["program"]
            if isinstance(spec, str):
                spec = compile_program(challenge, spec, base)
        if isinstance(spec, dict) and not _is_compiled(spec):
            spec = compile_program(challenge, spec)
        self.program = spec if isinstance(spec, CompiledProgram) else CompiledProgram(spec, name or "program")
        self.name = name or self.program.raw.get("meta", {}).get("title") or "program"
        resolved = []
        for e in entries:  # a strategy's own move paths resolve against its folder, extra entries' from the cwd / kit / repo
            e = dict(e)
            b = e.pop("_base", None)
            ref = e.get("move")
            if isinstance(ref, (str, Path)) and ref != EXTERNAL:
                from .moves import is_library_ref, knobs_of, resolve_move
                if e.get("host") == "raw" and e.get("knobs") is None and is_library_ref(str(ref)):
                    # a raw library move's command may use its card's knob names (sprint-raw@1: "speed"), as
                    # duel_env.opponent_spec attaches them; a strategy file given here directly had none
                    e["knobs"] = knobs_of(str(ref), resolve_move(str(ref)))
                e["move"] = load_runner(ref, b)
                if e.get("pin"):
                    from .moves import move_id_of
                    got = move_id_of(e["move"].path)
                    if got != e["pin"]:
                        raise ValueError(f"move {ref}: id {got}, the strategy pins {e['pin']}")
            resolved.append(e)
        self.moves = bind_moves(self.program, resolved, rt, load_runner)


def _is_compiled(p: dict) -> bool:
    return all("maxTime" in n for n in p.get("nodes", {}).values())


class DuelRow:
    """One judged row k (the state before tick k), by side (0 = A, 1 = B). Reused by its world: copy what you keep."""

    def __init__(self):
        self.k, self.t = 0, 0.0
        self.slot = np.zeros(2, np.int64)  # each side's duck slot (A in slot 0 unless swapped)
        self.obs = np.zeros((2, 61), np.float32)  # each side's policy obs for tick k (the command of its active node)
        self.priv = np.zeros((2, 6), np.float32)  # each side's critic extras (module doc)
        self.external = np.zeros(2, bool)  # the side's active node is EXTERNAL this tick: step() needs its action
        self.node = ["", ""]  # each side's active node id ("END" after one), its running behaviour (skill)
        self.skill = ["", ""]
        self.node_index = np.zeros(2, np.int64)  # frame node index (-1 = END)
        self.dist = np.zeros(2)  # trunk xy distance from the ring centre, m
        self.z = np.zeros(2)  # trunk height above the stage top, m
        self.tilt = np.zeros(2)  # trunk tilt, deg
        self.upright = np.zeros(2, bool)
        self.down = np.zeros(2, bool)  # the raw fall state (the judge's)
        self.down_rows = np.zeros(2, np.int64)
        self.radius = 0.0  # the safe circle this row, m
        self.touching = False  # a duck-duck contact in this row's state
        self.penetration = 0.0
        self.decided = False
        self.winner: int | None = None  # the winning side (None = a draw, or invalid)
        self.reason: str | None = None  # match.ts MatchEndReason
        self.t_end: float | None = None  # the decision row's time
        self.points = np.zeros(2)  # each side's points once decided
        self.crumbs = np.zeros(2, np.int64)  # Bread Fight: crumbs eaten so far, by side
        self.ate = np.zeros(2, bool)  # Bread Fight: the side ate a crumb on this row
        self.spots: list = []  # Bread Fight: the crumbs out (x, y) or None, by crumb (the world's list: copy to keep)
        self.invalid = False
        self.stopped = False  # no tick follows this row (the decision's tail is over, or the state broke)


class DuelWorld:
    """A duel Challenge's two-duck world (module doc). DuelWorld(a, b, challenge="sumo"): side A's and side B's specs (DuelSide;
    set later with set_side). reset(label, swapped=False, record=False) -> row 0; step(actions=None) -> the next row;
    advance(side) -> the next row where `side` acts (or the decision / the end); run() -> the last row; result()."""

    def __init__(self, a=None, b=None, challenge: str = "sumo", a_moves: list[dict] | None = None,
                 b_moves: list[dict] | None = None):
        self.challenge = challenge
        scene, _ = ensure_scene(challenge)
        self.info = info = load_info(challenge)
        if "duel" not in info:
            raise ValueError(f"Challenge {challenge} is not a duel (no duel block in its info)")
        self.duel, self.rules = info["duel"], info["duel"]["rules"]
        check_ported(info)
        self.rt = rt = MatchRuntime(info)
        self.m = m = world_model(scene)
        self.d = d = mujoco.MjData(m)
        if m.nq != info["nq"] or m.nv != info["nv"]:
            raise ValueError(f"world nq/nv {m.nq}/{m.nv} != the TS export's {info['nq']}/{info['nv']}")
        manifest = load_manifest(PLANT)
        self.ducks = [resolve_duck(m, manifest, i, w, rt.raw) for i, w in enumerate(self.duel["ducks"])]
        self.duck_nq = int(self.duel["duck_nq"])
        self.decimation = int(manifest["decimation"])
        self.stand_key = m.key("STAND").id
        cfg = BamConfig.from_block(manifest["actuator"])
        self.bams = []
        for k in self.ducks:
            bam = NumbaBam(m, k.qadr, k.vadr, k.act, cfg)
            bam.prev_torque = np.zeros(NUM_JOINTS)
            self.bams.append(bam)
        self.stock = {}
        for name in sorted(set(rt.policy[s] for s in rt.skill_names)):  # every skill's stock policy (kit 1.15.0)
            names = manifest["policies"][name]["command_names"]
            self.stock[name] = (load_runner(PLANT / "policies" / f"{name}.mlp.json"), "head_pose" in names,
                                "body_pose" in names)
        self.judge = MatchPoints(info, rt.policy_dt)
        # Wobbly Raft's swell (match.ts applySwell): a wave torque on the raft's ball joint; None = no swell (Sumo).
        # Log Roll v2 (LB-109): the same swell on a log's hinge (no raft in the scene): period 0 = a steady push in the
        # seeded sense, > 0 = it rocks the log
        scene = self.duel.get("scene") or []
        prop = next((p for p in scene if p["kind"] == "raft"), None) or next((p for p in scene if p["kind"] == "log"), None)
        fp = (prop or {}).get("params") or {}
        self.swell = None
        if prop is not None and float(fp.get("swell_nm", 0) or 0) > 0:
            hinge = prop["kind"] == "log"
            jname = f"{prop['id']}_{'hinge' if hinge else 'ball'}"
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if j < 0:
                raise ValueError(f"duel challenge {challenge}: no joint {jname} for the {prop['kind']}'s swell")
            self.swell = dict(dof=int(m.jnt_dofadr[j]), nm=float(fp["swell_nm"]), hinge=hinge,
                              start=float(fp.get("swell_from_s", 15 if hinge else 8)),
                              ramp=float(fp.get("swell_ramp_s", 10 if hinge else 15)),
                              period=float(fp.get("swell_period_s", 0 if hinge else 12)), phase=0.0, sense=1.0)
        # Bread Fight (kit 1.10.0): each crumb prop's free joint (match.ts: prop-drive.ts writes it every tick)
        bread = self.rules.get("bread") if self.rules.get("win") == "bread" else None
        self.bread, self.crumb_q, self.crumb_v, self.crumbs = bread, [], [], None
        for cid in crumb_ids(bread) if bread else []:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{cid}_free")
            if j < 0 or m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"duel challenge {challenge}: no free joint {cid}_free for the crumb {cid}")
            self.crumb_q.append(int(m.jnt_qposadr[j]))
            self.crumb_v.append(int(m.jnt_dofadr[j]))
        self.base = self.judge.base
        self.goal = self.duel["goal"]
        owner = np.full(m.ngeom, -1, np.int64)  # geom -> the slot whose duck it belongs to (-1: the stage, props)
        for k in self.ducks:
            owner[m.body_rootid[m.geom_bodyid] == m.body_rootid[k.trunk]] = k.slot
        self._owner = owner
        # per slot: views (persistent: MjData's buffers do not move) and loop state
        self._qv = [d.qpos[k.root_qpos_adr:] for k in self.ducks]
        self._vv = [d.qvel[k.root_dof_adr:] for k in self.ducks]
        self._qrel = [k.qadr - k.root_qpos_adr for k in self.ducks]
        self._vrel = [k.vadr - k.root_dof_adr for k in self.ducks]
        self._xquat = [d.xquat[k.trunk] for k in self.ducks]
        self._xmat, self._qpos = d.xmat, d.qpos
        self._obs = np.zeros((2, 61), np.float32)
        self._priv = np.zeros((2, 6), np.float32)
        self._vel = np.zeros((2, NUM_JOINTS))
        self._lag = np.zeros((2, NUM_JOINTS))
        self._head = np.zeros((2, 4))
        self._prev = np.zeros((2, NUM_JOINTS), np.float32)
        self._act = np.zeros((2, NUM_JOINTS), np.float32)
        self._tgt = np.zeros((2, NUM_JOINTS))
        self.sides: list[DuelSide | None] = [None, None]
        self._runtimes: list[DuckProgram | None] = [None, None]
        self.row = DuelRow()
        self.k = 0
        self._fused = None  # use_fused(): the fast rollout's fused substep loop (duck_rl/fused.py)
        # start_hook(world): called in reset after the draws (self.starts, self.slot_of set), before the poses are
        # applied: a training env draws the raw host's command there (duel_env.py); None = nothing
        self.start_hook = None
        for s, (spec, mv) in enumerate(((a, a_moves), (b, b_moves))):
            if spec is not None:
                self.set_side(s, spec, mv)

    def set_side(self, side: int, spec, moves: list[dict] | None = None, name: str | None = None) -> DuelSide:
        """Side `side` (0 = A, 1 = B) plays `spec` (DuelSide) from the next reset on."""
        ds = spec if isinstance(spec, DuelSide) else DuelSide(self.challenge, self.rt, spec, moves, name)
        self.sides[side] = ds
        self._runtimes[side] = DuckProgram(self.rt, ds.program, self.stock, ds.moves, self.info.get("routes"))  # own state even for one DuelSide on both
        return ds

    # ---- reset (match.ts:323-357 and World's resets) ----
    def _reset_state(self, init: bool) -> None:
        """World.reset: plant.ts resetToStand for both ducks (the STAND keyframe of the whole data, then each duck's
        servo qpos and ctrl = DEFAULT_POS, qvel = 0), `init` (the spawn poses), each duck's loop and BAM state."""
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, self.stand_key)
        for k in self.ducks:
            d.qpos[k.qadr] = k.default_pos
            d.ctrl[k.act] = k.default_pos
        d.qvel[:] = 0.0
        if init:
            q, q0 = d.qpos, m.qpos0
            q[self.duck_nq:] = q0[self.duck_nq:]  # props start at their MJCF pose
            for slot, k in enumerate(self.ducks):
                a = k.root_qpos_adr
                st = self.starts[slot]
                q[a], q[a + 1] = st["x"], st["y"]
                z = self.rules["spawns"][slot].get("z")
                if z is not None:
                    q[a + 2] = q[a + 2] + z
                q[a + 3:a + 7] = [math.cos(st["yaw"] / 2), 0.0, 0.0, math.sin(st["yaw"] / 2)]
            if self.crumbs is not None:  # match.ts drives.apply(data, 0) in the reset
                self._drive_crumbs(0.0)
        self._prev[:] = 0.0
        self._head[:] = 0.0
        self._lag[:] = 0.0
        self._has_lag = [False, False]
        for k, bam in zip(self.ducks, self.bams):
            bam.reset(d, k.default_pos)

    def reset(self, label: str, swapped: bool = False, record: bool = False) -> DuelRow:
        """A new episode on PRNG label base `label` ("sumo:3:<seed>" = TS public seed <seed>; module doc), side A in slot
        0 (swapped: slot 1). record: keep every row's qpos, ring radius and each slot's node / skill (frames(), and
        the trajectory hash) and the events. Returns row 0."""
        self._begin(label, swapped, record)
        return self._observe()

    def replay(self, qpos_rows, label: str, swapped: bool = False) -> DuelRow:
        """The judge and both programs on given states instead of this world's physics (the logic check of
        tools/duel_parity.py): the episode's reset (its draws, so the noise streams are the episode's), then per row r
        qpos := qpos_rows[r] (e.g. a TS match's float32 frames), qvel := 0 (the gyro then reads 0, which only the walk
        command's heading hold and the yaw_rate signal see), mj_forward, and row r as in an episode, until the episode
        stops (or the rows run out). Recorded as reset(record=True); returns the last row."""
        self._begin(label, swapped, True)
        d = self.d
        row = self.row
        for r, q in enumerate(qpos_rows):
            d.qpos[:] = q
            d.qvel[:] = 0.0
            self.k = r
            self.rec_qpos[r:] = [d.qpos.copy()]
            row = self._observe()
            if row.stopped:
                break
        return row

    def _begin(self, label: str, swapped: bool, record: bool) -> None:
        if self.sides[0] is None or self.sides[1] is None:
            raise ValueError("both sides need a program (set_side) before reset")
        m, rt, rules = self.m, self.rt, self.rules
        self.slot_of = (1, 0) if swapped else (0, 1)  # side -> slot
        self.side_of = (1, 0) if swapped else (0, 1)  # slot -> side
        self.label, self.swapped = label, swapped
        prngs = [match_prng(label, 0), match_prng(label, 1)]
        dr = [draw_randomisation(prngs[0], rt.dr), draw_randomisation(prngs[1], rt.dr)]
        pm = match_prng(label, "match")  # the shared stream: the spawn-line rotation, then (only with one) the swell
        theta = pm.uniform(-math.pi, math.pi) if rules["rotate_spawns"] else 0.0
        if self.swell is not None:  # match.ts: phase uniform(-pi, pi), then sense uniform(0, 1) < 0.5 ? -1 : 1
            self.swell["phase"] = pm.uniform(-math.pi, math.pi)
            self.swell["sense"] = -1.0 if pm.uniform(0.0, 1.0) < 0.5 else 1.0
        cx, cy = float(rules["ring"]["x"]), float(rules["ring"]["y"])
        # Bread Fight: the crumb field on the shared stream (match.ts makes it after the rotation; it draws when placing)
        self.crumbs = None if self.bread is None else CrumbField(self.bread, 2, (cx, cy), pm)
        ct, st = math.cos(theta), math.sin(theta)
        self.draws, self.theta, self.starts = dr, theta, []
        for slot, k in enumerate(self.ducks):
            for g in k.soles:
                m.geom_friction[g, 0] = dr[slot]["friction"]
            m.body_mass[k.trunk] = k.base_mass * dr[slot]["mass"]
            m.body_inertia[k.trunk] = k.base_inertia * dr[slot]["mass"]
            s = rules["spawns"][slot]
            dx, dy = s["x"] - cx, s["y"] - cy
            self.starts.append(dict(x=cx + ct * dx - st * dy + dr[slot]["dx"], y=cy + st * dx + ct * dy + dr[slot]["dy"],
                                    yaw=s["yaw"] + theta + dr[slot]["dyaw"], friction=dr[slot]["friction"],
                                    mass=dr[slot]["mass"]))
        if self.start_hook is not None:
            self.start_hook(self)
        self._reset_state(init=False)  # the World constructor's reset, then mj_setConst on the drawn model
        mujoco.mj_setConst(m, self.d)
        self._reset_state(init=True)
        self.programs = [self._runtimes[self.side_of[slot]] for slot in (0, 1)]  # by slot
        for slot, k in enumerate(self.ducks):
            self.programs[slot].reset(m, k, prngs[slot], self.ducks[1 - slot].head_body, self.goal, self.base,
                                      rival=(self.ducks[1 - slot].trunk, slot),  # match.ts rivalOf (1v1: role = slot)
                                      crumbs=None if self.crumbs is None else CrumbSense(self.crumbs, self.judge, slot))
        self.judge.reset(self.crumbs)
        self.k = 0
        self.record = record
        self.rec_qpos = [self.d.qpos.copy()] if record else None  # row 0 = the reset state (the World's row 0)
        self.rec_ring, self.rec_node, self.rec_skill = [], [], []
        self.rec_crumbs = []  # Bread Fight, recorded: per row r the crumb poses written for row r + 1 (frames())

    # ---- one row (match.ts:418-485) ----
    def _unstable(self) -> bool:
        """MuJoCo instability warnings so far in the episode (match.ts unstable(); mj_resetData clears them)."""
        w = self.d.warning
        return sum(w[i].number for i in WARNINGS) > 0

    def _observe(self) -> DuelRow:
        m, d, k, rt, judge, row = self.m, self.d, self.k, self.rt, self.judge, self.row
        mujoco.mj_forward(m, d)
        q, xm = self._qpos, self._xmat
        row.k, row.t = k, k / rt.ticks_per_s
        finite = all(math.isfinite(q[a] + q[a + 1] + q[a + 2] + q[a + 3]) for a in (x.root_qpos_adr for x in self.ducks))
        if not finite or (judge.decided and self._unstable()):
            if not judge.decided:
                judge.invalidate(k)  # this row is not a valid state
            self.rows = k
            return self._fill(stopped=True)
        cos = (float(xm[self.ducks[0].trunk, 8]), float(xm[self.ducks[1].trunk, 8]))
        for slot in (0, 1):
            self.programs[slot].observe_row(cos[slot])
            a = self.ducks[slot].root_qpos_adr
            self.programs[slot].x, self.programs[slot].y = float(q[a]), float(q[a + 1])  # match.ts observeRow: the routes' pose
        touch = judge.needs_touch and _contacts(d.contact.geom, d.ncon, self._owner, d.efc_type, d.efc_pos, d.nefc)[0]
        heads = None if self.crumbs is None else [d.xpos[x.head_body, :2] for x in self.ducks]
        judge.push(q, None, cos_tilt=cos, unstable=self._unstable, touching=bool(touch), heads=heads)
        self.rows = k + 1
        if self.record:
            self.rec_ring.append(judge.radius)
            if self.crumbs is not None:
                t1 = (k + 1) / rt.ticks_per_s
                self.rec_crumbs.append([self.crumbs.pose(j, t1) for j in range(self.crumbs.k)])
        if judge.invalid or k >= judge.stop_row:
            return self._fill(stopped=True)
        for slot in (0, 1):
            self.programs[slot].step(k, d)
        for slot, x in enumerate(self.ducks):
            p = self.programs[slot]
            has = self._has_lag[slot]
            nb_obs(self._obs[slot], self._priv[slot], self._vel[slot], self._head[slot], d.sensordata, x.gyro_adr,
                   self._xquat[slot], self._qv[slot], self._vv[slot], self._qrel[slot], self._vrel[slot], x.default_pos,
                   self._lag[slot] if has else self._vel[slot], has, self._prev[slot], p.command, p.head_in_obs,
                   p.body_in_obs, self.draws[slot]["friction"], self.draws[slot]["mass"])
            self._priv[slot, 3] = 10 * ((q[x.root_qpos_adr + 2] - self.base) - STAND_Z)  # height from the stage top
        if self.record:
            self.rec_node.append([self.programs[s].node_index for s in (0, 1)])
            self.rec_skill.append([self.programs[s].skill_index for s in (0, 1)])
        return self._fill(stopped=False)

    def _fill(self, stopped: bool) -> DuelRow:
        row, judge, d = self.row, self.judge, self.d
        row.stopped = stopped
        touch, pen = _contacts(d.contact.geom, d.ncon, self._owner, d.efc_type, d.efc_pos, d.nefc)
        row.touching, row.penetration = bool(touch), float(pen)
        row.radius = judge.radius
        row.decided, row.invalid, row.reason, row.t_end = judge.decided, judge.invalid, judge.reason, judge.t
        row.winner = None if judge.winner is None else self.side_of[judge.winner]
        cf = self.crumbs
        if cf is not None:
            row.spots = cf.spots
            for side in (0, 1):
                slot = self.slot_of[side]
                row.crumbs[side], row.ate[side] = cf.counts[slot], judge.ate == slot
        pts = judge.points
        for side in (0, 1):
            slot = self.slot_of[side]
            p = self.programs[slot]
            row.slot[side] = slot
            row.obs[side] = self._obs[slot]
            row.priv[side] = self._priv[slot]
            row.external[side] = (not stopped) and p.external
            row.node[side] = p.node_name
            row.skill[side] = p.layer.running
            row.node_index[side] = p.node_index
            row.dist[side], row.z[side] = judge.dist[slot], judge.z[slot] - self.base
            c = judge.cos_tilt[slot]
            row.tilt[side] = math.acos(max(-1.0, min(1.0, c))) * 180 / math.pi
            row.upright[side], row.down[side] = judge.upright[slot], judge.down[slot]
            row.down_rows[side] = judge.down_rows[slot] if judge.down[slot] else 0
            row.points[side] = pts[slot]
        return row

    def step(self, actions=None) -> DuelRow:
        """Tick k: each slot's action (its runner on the row's obs, or for an EXTERNAL tick actions[side]: a float32[14]
        policy output; `actions` = {side: action} or a sequence by side), targets = DEFAULT_POS + action, both BAMs
        latch, decimation x (BAM slot 0, BAM slot 1, mj_step); then row k + 1."""
        if self.row.stopped:
            raise RuntimeError("the episode is over (row.stopped): reset first")
        m, d = self.m, self.d
        for slot, x in enumerate(self.ducks):
            p = self.programs[slot]
            a = self._act[slot]
            if p.external:
                side = self.side_of[slot]
                ext = None if actions is None else (actions.get(side) if isinstance(actions, dict) else actions[side])
                if ext is None:
                    raise ValueError(f"side {'AB'[side]} is on EXTERNAL node {p.node_name!r} at tick {self.k}: step() needs its action")
                a[:] = np.asarray(ext, np.float32)
            else:
                p.runner.run(self._obs[slot], a)
            head = p.head
            if head is not None:  # sim.ts DuckLoop.holdHead: the held head joints (actions 5-8) approached at HEAD_EMA
                prev = self._prev[slot]
                for i in range(4):
                    v = head[i]
                    if math.isfinite(v):
                        pv = float(prev[5 + i])
                        a[5 + i] = pv + self.rt.head_ema * (v - pv)
            np.add(x.default_pos, a, out=self._tgt[slot])
            self._prev[slot] = a
            self._lag[slot] = self._vel[slot]
            self._has_lag[slot] = True
            self.bams[slot].latch(self._tgt[slot])
        if self.swell is not None:  # match.ts applySwell(data, k / TICKS_PER_S), before the tick's substeps
            sw, t = self.swell, self.k / self.rt.ticks_per_s
            a = sw["nm"] * min(1.0, max(0.0, (t - sw["start"]) / sw["ramp"]))
            f = d.qfrc_applied
            if sw["hinge"]:  # the log: one dof
                f[sw["dof"]] = (a * math.cos(sw["phase"] + (sw["sense"] * 2 * math.pi * t) / sw["period"])
                                if sw["period"] > 0 else a * sw["sense"])
            else:
                ang = sw["phase"] + (sw["sense"] * 2 * math.pi * t) / sw["period"]
                f[sw["dof"]], f[sw["dof"] + 1], f[sw["dof"] + 2] = a * math.cos(ang), a * math.sin(ang), 0.0
        if self.crumbs is not None:  # match.ts drives.apply(data, (k + 1) / TICKS_PER_S), before the tick's substeps
            self._drive_crumbs((self.k + 1) / self.rt.ticks_per_s)
        b0, b1 = self.bams
        if not (self._fused is not None and self._fused.ready(m, d) and self._fused.run(b0, b1)):  # (fused: one call)
            for _ in range(self.decimation):
                b0.before_substep(d)
                b1.before_substep(d)
                mujoco.mj_step(m, d)
        self.k += 1
        if self.record:
            self.rec_qpos.append(d.qpos.copy())
        return self._observe()

    def _drive_crumbs(self, t: float) -> None:
        """prop-drive.ts apply: each crumb prop's free-joint qpos for time t, its qvel zeroed."""
        q, v = self.d.qpos, self.d.qvel
        for j, (a, b) in enumerate(zip(self.crumb_q, self.crumb_v)):
            q[a:a + 7] = self.crumbs.pose(j, t)
            v[b:b + 6] = 0.0

    def use_fused(self) -> bool:
        """Run step()'s substeps in duck_rl/fused.py's kernel (bit-identical; the training env's fast rollout). False
        when the kernel is not available here (the stock loop stays)."""
        from .fused import duel_tick
        self._fused = duel_tick(self)
        return self._fused is not None

    def advance(self, side: int = 0, stop_at_decision: bool = True) -> DuelRow:
        """Steps with the sides' own runners until a row where `side` acts (row.external[side]), the decision (unless
        stop_at_decision is False: then the end of its tail) or the end. Returns that row (maybe the current one)."""
        row = self.row
        while not row.stopped and not row.external[side] and not (stop_at_decision and row.decided):
            row = self.step()
        return row

    def run(self) -> DuelRow:
        """Steps to the end of the episode (the decision's tail, or a broken state); every side must run internally."""
        row = self.row
        while not row.stopped:
            row = self.step()
        return row

    # ---- results ----
    def result(self) -> dict:
        """The episode as match.ts reports it (scripts/match.ts --json rows, by side): winner "A" | "B" | "draw" |
        "invalid", reason, t, points by side, the slot A played, each slot's start and draws."""
        j = self.judge
        if not j.decided:
            raise RuntimeError("no decision yet")
        win = "invalid" if j.invalid else "draw" if j.winner is None else "AB"[self.side_of[j.winner]]
        out = dict(label=self.label, a_slot=self.slot_of[0], winner=win, reason=j.reason, t=j.t,
                   a_points=j.points[self.slot_of[0]], b_points=j.points[self.slot_of[1]], rows=self.rows,
                   starts=self.starts, theta=self.theta)
        if self.crumbs is not None:  # match.ts sideResult crumbs / crumb_t, by side
            out.update(crumbs=[self.crumbs.counts[self.slot_of[s]] for s in (0, 1)],
                       crumb_t=[list(j.crumb_t[self.slot_of[s]]) for s in (0, 1)])
        return out

    def frames(self) -> dict:
        """Recorded rows (reset(record=True)): qpos [rows, nq] float64 (row r = the state before tick r), ring [rows],
        node / skill [rows, 2] by slot (row r: tick r - 1's, row 0: tick 0's, as match.ts frames), the trajectory hash
        (sha256 over the float64 qpos rows, match.ts trajSha256) and each slot's events; Bread Fight: crumbs [rows - 1,
        n crumbs, 7], row r = the poses written after row r's update (TS frame row r + 1's crumb qpos)."""
        if not self.record:
            raise RuntimeError("reset(..., record=True) to record frames")
        n = self.rows
        qpos = np.asarray(self.rec_qpos[:n])
        node = np.asarray(self.rec_node, np.int64).reshape(-1, 2)
        skill = np.asarray(self.rec_skill, np.int64).reshape(-1, 2)
        ticks = np.maximum(0, np.arange(n) - 1)
        return dict(qpos=qpos, ring=np.asarray(self.rec_ring[:n]), node=node[np.minimum(ticks, len(node) - 1)],
                    skill=skill[np.minimum(ticks, len(skill) - 1)],
                    traj_sha256=hashlib.sha256(np.ascontiguousarray(qpos, "<f8").tobytes()).hexdigest(),
                    events=[self.programs[s].events for s in (0, 1)],
                    **({"crumbs": np.asarray(self.rec_crumbs[:n])} if self.crumbs is not None else {}))
