"""The training env for any Challenge: lab.fastenv.FastDuckEnv on the Challenge's world (buildWorldXml, exported by
ts/prepare.ts), with the TS skills-layer command of the trained node(s), the Challenge metric online (duck_rl/metrics.py,
per metric kind), the floor-out rule, the Challenge's prop state (duck_rl/challenges/<id>.py), spawn or handoff starts. Ported
from the disc-rl kit's DiscEnv (itself from the wheel kit's WheelEnv), made Challenge-generic.

  * command: the trained node's (walk: vx/vy/wz with the start kick and heading hold, head params; stand: head params),
    fed every tick as TS does (controller.WalkController); obs = alpha_walking's 61-dim obs (the stock policy of walk and
    stand nodes; the TS eval builds exactly this obs, so do not add inputs to the actor);
  * raw host (cfg host "raw", the sim's NodePolicy host "raw"): the node's own 13-value block raw_command[node id] fed
    verbatim (controller.RawController: no speed map, kick or yaw loop), head and body in the obs, the head slots
    through the obs builder's head EMA as in TS; command_ranges {slot index: (lo, hi)} redraws those slots uniformly at
    every training reset (one draw per episode, used on every trained node; command-conditioned training); evaluation
    resets (explicit draws or a bank state given) feed raw_command as it is;
  * done (cfg done {after_s?, upright_s?, rolled_then_upright_s?}, the move's own rule, skills.ts SkillLayer #account):
    the `move_done` column turns 1 when a rule fires, counted in the trained node's ticks since its entry (the reset,
    or a schedule switch); end_on_done ends the episode there as a termination (no bootstrap: what follows belongs to
    the duckscript's next node, which the env cannot run; pay for the exit state in the reward if it matters);
  * schedule (optional): [(t, node)] walk -> walk switches at fixed times (the program's `t >= T` guards; one policy on
    several nodes): the controller is rebuilt with no start kick, as TS SkillLayer.enter does for a moving duck;
  * critic priv[4:6] = [the first hinged prop's joint velocity (0 if none), t / duration];
  * metric: the Challenge's own (packages/sim/src/runtime/metrics.ts), one row per tick, row 0 at reset, with the TS
    zeroing after an out (`metric` = the TS trajectory's metric column); `success` = the Challenge's bar once settled;
  * out rule (episode.ts floor_out_before_s, when the Challenge has one): any contact between the floor geom and a geom of
    the duck's tree, checked after every physics substep and at each row (the TS check sees the same contacts); out
    before the limit = metric 0 (`lost`); cfg out_before_s moves the limit (training only: a safety margin);
  * resets: spawn (the Challenge's reset exactly as episode.ts: spawn x/y/yaw + the draws dx, dy, dyaw, sole friction,
    trunk mass, spawn z; joint noise in training) or handoff (a bank state ts/eval.ts --bank recorded where the program
    first enters the trained node, with that seed's draws, loop state and metric tail);
  * training-only randomisation: the prop's scales (challenges/<id>.py), the actuator (supply voltage, kt, R);
  * termination: fall rule ("tilt45" | "down" | "none"), optional max_tilt_deg, end_on_out (default: an out before the
    limit ends the episode), end_on_finish (a race: end on the row the metric first meets the Challenge's bar, the TS
    `success` event = the finish; the metric mirror's `finished` / `finish_t` columns), end_s (default: the adapter's,
    e.g. the bell on the turntable, else the Challenge's duration), term_at_end (the end is a true termination: no time-out
    bootstrap). `end_reason` after an end: "fall", "finish", "out", "done", "stop" (end_on_stop, end_after_out_s) or
    "timeout" (the episode's end, also with term_at_end); the workers report it per episode (kit 1.6.2);
  * stagger_start (kit 1.6.2, train.py STAGGER_START): each env's first training episode is cut at a uniformly drawn
    length (1 tick .. the full episode; a time-out, bootstrapped even with term_at_end), so the envs run out of phase;
  * mechanics (kit 1.9.0, duck_rl/movers.py; only a Challenge whose info names one: Crossy Duck's `traffic`, Red Light's
    `light`): the runtime's
    prop drives written at the reset (t = 0) and before every tick (t = (k + 1) / 50), drawn per episode from the
    "<label>:match" stream (label: an explicit reset's ep["label"] or ep["seed"], the public seed's episode label; else a
    training label "<challenge>:<version>:train:<env seed>:<n>"), and the mover's out rule checked on every row before the
    finish (a car hit: `hit`; moving on red: `red_out`; it zeroes the metric like an out before the limit, so `lost`, the race's miss for the
    finish columns, and with end_on_out the episode ends, reason "out"). A Challenge without one runs the old code exactly;
  * program mode (kit 1.9.0, cfg program = the duckscript compiled in full (ts/prepare.ts --program-full) +
    program_nodes = the trained nodes; train.py RUN_PROGRAM = True): the env runs the whole duckscript as episode.ts
    does (program.py DuckProgram: skills layer, 10-Hz program ticks, SoloSensors with the movers' senses and maze_yaw);
    the policy acts on the ticks a trained node runs, every other node runs its stock skill (or, with base_moves, the move
    bound to it: BASE_STRATEGY, kit 1.17.0) inside the step until the
    next trained tick (one transition; the row returned is that tick's, with `ticks` = the ticks it covered and
    `dmetric` / `finish_now` over all of them, so a config can pay the waited time and the metric gained while stock
    nodes ran: duck_rl/movers.py's Challenges only, program_columns), and entering a node of skip_end_skills
    (get_up: the duckscript's answer to a fall) ends the episode as a fall. Spawn starts only (kit 1.17.0: the start node need not be
    trained: the reset runs the untrained nodes until a trained one runs); no schedule, bank, raw host, move done rule or `progress` guard. Needs the info's runtime block (a solo
    Challenge with a ported mechanic exports it).

cfg (JSON-able, all optional but node and info): node {skill, params, id?}, schedule [(t, node)], start "spawn"|"handoff",
program (compiled), program_nodes [node ids], base_moves [duck-strategy/1 move entries: the other nodes' moves], skip_end_skills (["get_up"]),
host ("walk-stand" | "raw"), raw_command {node id: 13 floats} (raw: required for every trained node), command_ranges
{slot index: (lo, hi)}, done {after_s?, upright_s?, rolled_then_upright_s?}, end_on_done,
bank (path), handoff_share, end_s, term_at_end, fall, down_s, max_tilt_deg, end_on_out, end_on_stop (end when a metric
that can freeze has frozen: disc_turns), end_on_finish (end at the finish: the first row that meets the Challenge's bar),
out_before_s, act_rand,
joint_noise, scene, info, plus the adapter's keys (disc_scale/disc_rpm0, wheel_scale/wheel_rpm0, prop_scale/prop_rpm0).
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
import multiprocessing as mp
import os
import re
from pathlib import Path

import mujoco
import numpy as np
from numba import njit

from . import PLANT
from .controller import RawController, WalkController
from .challenges import adapter_columns, make_adapter
from .metrics import MIRRORS, NoMirror, make_metric, meets_success
from .movers import TICKS_PER_S, make_mechanics, mover_columns
from .prng import draw_randomisation, episode_label, label_prng, rng_key
from duckref.plant import physics_only, with_camera_geoms  # packages/sim/src/plant.ts physicsOnly (the world and the plant alike)
from lab.duck import DT, LAYOUT, STATE_DIM, Tracker, apply_episode, draw_episode, is_fall, make_duck
from lab.duck import unpack as lab_unpack
from lab.fastenv import FastDuckEnv, nb_obs, nb_state
from lab.pool import Pool

PUSH_OFF = dict(first_s=(1e9, 1e9), every_s=(1.0, 1.0), speed=(0.0, 0.0))
BASE = ["x", "y", "hdg", "up32", "ended", "out", "lost", "metric", "dmetric", "settled", "success",
        "copper", "mech", "oor", "atlim", "move_done"]
FELL_I, TERM_I = LAYOUT["fell"][0], LAYOUT["terminated"][0]
COS_UPRIGHT = math.cos(math.radians(32.0))
LEGS = np.r_[0:5, 9:14].astype(np.int64)  # plant joint order: left leg 0-4, head 5-8, right leg 9-13
ACT_RAND = dict(vin=(6.5, 7.6), kt=(0.95, 1.05), R=(0.95, 1.05))  # NumbaBam.P[6], P[0], P[1] (nominal 7.4 V)
OBS_POLICY = "alpha_walking"
VISION_EVERY = 5  # rows per program tick (10 Hz): when a camera move's picture refreshes, as in packages/sim Sensors
_BANKS: dict[str, list] = {}
_WORLDS: dict[str, mujoco.MjModel] = {}

BASE_DOC = {
    "x, y": "trunk position (m, world)", "hdg": "heading (rad)", "up32": "1 if tilt < 32 deg",
    "ended": "1 on the tick the episode ended by the out rule (not a fall; always 0 on a Challenge without floor_out_before_s)",
    "out": "1 once any duck geom touched the floor (any time; Challenges with floor_out_before_s only)",
    "lost": "1 once out before the limit (floor_out_before_s, or cfg out_before_s): the metric is 0 from then on",
    "metric": "the Challenge metric so far, as the TS trajectory's metric column (0 once lost; NaN for a kind without a mirror)",
    "dmetric": "metric change this tick", "settled": "1 once the metric may declare success (TS settled())",
    "success": "1 once settled and the metric meets the Challenge's bar (the TS `success`)",
    "copper": "leg copper loss (W)", "mech": "leg positive mechanical power (W)", "oor": "leg targets out of range (rad)",
    "atlim": "share of substeps with a leg servo at its current limit",
    "move_done": "1 once the move's own done rule (cfg done) fired on the current node (always 0 without one)",
    "(lab columns)": "t, tilt, upright, height, lin_vel, ang_vel, gyro, gravity, joint_pos, joint_vel, action, "
                     "prev_action, command, feet_contact, feet_z, fell, terminated, ... (lab/duck.py LAYOUT)",
}


def columns(info: dict) -> dict[str, int]:
    """State column index by name (lab columns come from lab.duck.unpack): the generic ones, the metric mirror's, the
    Challenge adapter's, and the aliases (the same index as their column). A duel Challenge (info `duel`, duck_rl/duel_env.py):
    the adapter's columns only, from the learner's side."""
    mcls = MIRRORS.get(info["metric"]["kind"], NoMirror)
    acols, aal, _ = adapter_columns(info)
    if "duel" in info:
        names, aliases = list(acols), dict(aal)
    else:
        mvcols = mover_columns(info)[0] + program_columns(info)[0]  # [] without a mechanic: the old layout
        names, aliases = BASE + [c for c in mcls.COLUMNS + acols + mvcols if c not in BASE], {**mcls.ALIASES, **aal}
    ex = {n: STATE_DIM + i for i, n in enumerate(names)}
    for a, c in aliases.items():
        ex[a] = ex[c]
    return ex


def state_dim(info: dict) -> int:
    return STATE_DIM + len(set(columns(info).values()))


def state_doc(info: dict) -> dict:
    if "duel" in info:
        return {"(lab columns)": BASE_DOC["(lab columns)"] + " (of the learner's duck; height, head_z and feet_z from "
                "the floor, not the stage top: the Challenge's z columns, e.g. Sumo's z_self, are from the top)",
                **adapter_columns(info)[2]}
    mcls = MIRRORS.get(info["metric"]["kind"], NoMirror)
    return {**BASE_DOC, **mcls.DOC, **adapter_columns(info)[2], **mover_columns(info)[1], **program_columns(info)[1]}


def program_columns(info: dict) -> tuple[list[str], dict]:
    """The program-mode columns (module doc), on a solo Challenge whose info has the runtime block (ts/prepare.ts exports it
    only for a Challenge with a ported mechanic); ([], {}) else (the old layout)."""
    if "runtime" not in info or "duel" in info:
        return [], {}
    return ["ticks"], {"ticks": "the 50-Hz ticks this row's transition covered: 1, or in program mode 1 + the ticks the "
                                "untrained nodes ran stock inside the step (their time; dmetric and finish_now then "
                                "cover the whole step too)"}


def make_unpack(info: dict):
    """unpack(S[N, D]) -> {column: S[:, i]} over the lab's columns and this Challenge's (train.py hands it to lab.learn)."""
    ex = columns(info)

    def unpack(S: np.ndarray) -> dict:
        out = lab_unpack(S)
        for n, i in ex.items():
            out[n] = S[:, i]
        return out
    return unpack


def load_bank(path: str) -> list:
    if path not in _BANKS:
        _BANKS[path] = json.loads(Path(path).read_text())["states"]
    return _BANKS[path]


def world_model(scene: str, camera: bool = False) -> mujoco.MjModel:
    """A fresh copy of the compiled world (loaded once per process; each duck owns a copy: the BAM and the draws write
    into the model). Kit 1.16.0: the wasm-compiled world ts/prepare.ts writes beside the scene (<scene stem>.mjb, the
    server's own compile, physics only), so the model arrays are the sim's byte for byte; a native compile of the XML
    builds other mesh polygon tables (mm contact jumps on boxes, discs, stages). DUCK_RL_WORLD=xml forces the native
    compile (physics only, as before 1.16.0) for comparisons; a missing .mjb falls back to it with a warning. `camera`:
    the world a palette camera move's run compiles (<stem>.camera.mjb; duckref.plant with_camera_geoms: the camera
    geoms, docs/vision/PALETTE.md); else the plain world, main's model."""
    key = (scene, camera)
    if key not in _WORLDS:
        mjb = Path(scene).with_suffix(".camera.mjb" if camera else ".mjb")
        native = os.environ.get("DUCK_RL_WORLD", "mjb") == "xml"
        why = None if native else f"no {mjb.name} beside {scene}"
        if not native and mjb.exists():
            try:
                _WORLDS[key] = mujoco.MjModel.from_binary_path(str(mjb))
            except Exception as e:  # an MJB of another MuJoCo version (the kit pins mujoco 3.14.0, as @mujoco/mujoco)
                why = f"{mjb.name} does not load in mujoco {mujoco.__version__} ({e})"
        if key not in _WORLDS:
            if why is not None:
                print(f"[duck-rl] warning: {why}: compiling the XML natively (contacts drift from the server's sim; "
                      f"duck_rl.ensure_scene(challenge, force=True) exports it)", flush=True)
            text = Path(scene).read_text()
            xml = physics_only(with_camera_geoms(text) if camera else text)
            xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{PLANT}/meshes/"', xml, count=1)
            _WORLDS[key] = mujoco.MjModel.from_xml_string(xml)
    return copy.copy(_WORLDS[key])


def heading(xmat) -> float:
    return math.atan2(-xmat[1], xmat[4])


@njit(cache=True)
def _effort(acc, tau, qvel, legs, vadr_legs, tlim):
    """acc += [sum tau^2, sum max(0, tau qdot), any |tau| >= tlim] over the leg joints (one substep)."""
    at = 0.0
    for i in range(legs.shape[0]):
        t = tau[legs[i]]
        acc[0] += t * t
        p = t * qvel[vadr_legs[i]]
        if p > 0.0:
            acc[1] += p
        if abs(t) >= tlim:
            at = 1.0
    acc[2] += at


@njit(cache=True)
def _floor_touch(cgeom, ncon, floor, duck_mask) -> bool:
    for c in range(ncon):
        g1, g2 = cgeom[c, 0], cgeom[c, 1]
        if (g1 == floor and duck_mask[g2]) or (g2 == floor and duck_mask[g1]):
            return True
    return False


def make_controller(node: dict) -> WalkController:
    p = dict(node.get("params") or {})
    if node["skill"] == "walk":
        return WalkController(vx=p.get("vx", 0.0), vy=p.get("vy", 0.0), wz=p.get("wz", 0.0),
                              head_pitch=p.get("head_pitch", 0.0), head_yaw=p.get("head_yaw", 0.0),
                              head_roll=p.get("head_roll", 0.0))
    if node["skill"] == "stand":
        return WalkController(vx=0.0, vy=0.0, wz=0.0, head_pitch=p.get("head_pitch", 0.0),
                              head_yaw=p.get("head_yaw", 0.0), head_roll=p.get("head_roll", 0.0), walk=False)
    raise ValueError(f"trained node skill {node['skill']!r} (walk or stand only)")


class ChallengeEnv(FastDuckEnv):
    PRIV = 6

    def __init__(self, cfg: dict, seed: int, fused: bool = False):
        self.cfg = cfg
        self.info = info = cfg["info"]
        # a frozen camera encoder reads the palette picture: the world with the camera geoms (as the site's run of the export)
        self.duck = duck = make_duck("numba", model=world_model(str(cfg["scene"]), camera=bool(cfg.get("encoder"))))
        self.policy = OBS_POLICY
        self.lo = self.hi = np.zeros(13)
        self.challenge_s = float(info["duration_s"])
        self.rules, self.push = {}, PUSH_OFF
        self.noise = float(cfg.get("joint_noise", 0.03))
        self.rng = np.random.default_rng(seed)
        names = duck.command_names(self.policy)
        self.host = cfg.get("host") or "walk-stand"
        if self.host not in ("walk-stand", "raw"):
            raise ValueError(f"host {self.host!r}: 'walk-stand' or 'raw'")
        self.host_raw = self.host == "raw"
        # the raw host feeds head and body to every move (skills.ts #raw); alpha_walking's obs has both anyway
        self._use_head = "head_pose" in names or self.host_raw
        self._use_body = "body_pose" in names or self.host_raw
        self._ranges = {int(i): (float(lo), float(hi)) for i, (lo, hi) in (cfg.get("command_ranges") or {}).items()}
        self.cmd_stage = cfg.get("cmd_stage")  # E2.1: {height: m, vx: m/s} -> switch the walk command once per episode
        self.cmd_sched = cfg.get("cmd_schedule")  # T2': episode | time | state command scheduling for the walk host
        self._cmd = None
        self._draw: dict[int, float] = {}
        done = cfg.get("done") or {}
        self.done_ticks = {k: int(round(float(done[k]) / DT)) for k in ("after_s", "upright_s", "rolled_then_upright_s")
                           if done.get(k) is not None}
        self.end_on_done = bool(cfg.get("end_on_done", False))
        self._soles = np.asarray(duck.soles, np.int64)
        self._qadr, self._vadr = np.asarray(duck.qadr, np.int64), np.asarray(duck.vadr, np.int64)
        self._ob, self._pr, self._vn = np.empty(61, np.float32), np.empty(6, np.float32), np.empty(14)
        # the ray eye's VISION mode (cfg "vision": the obs gained its 12 numbers) was removed with the eye (LB-330)
        if cfg.get("vision"):
            raise ValueError("cfg vision: the ray eye was removed (LB-330); a move that sees is a camera move (ENCODER)")
        # a frozen camera encoder (cfg "encoder", train.py ENCODER, kit 1.27.0): the site camera move's picture (the camera's
        # rays in the palette look, vision.py RaySight) through the encoder (encoder.py): the obs gains its `out` named
        # numbers, refreshed every program tick (VISION_EVERY rows) as the site refreshes a camera move's picture.
        # enc_defer (set by the pool's workers): the forward waits for flush_encoders, batched over the worker's envs
        self.cam = self.encoder = None
        self.enc_defer = self.enc_dirty = self._view_stale = False
        if cfg.get("encoder"):
            from .encoder import load as load_encoder
            from .vision import PALETTE, PICTURE_SHAPE, RaySight
            self.encoder = load_encoder(cfg["encoder"])
            self.cam = RaySight(duck.model)
            # before the first refresh, as the site's Vision (vision.ts): depth 1, colours the sky (a ray's miss)
            self.picture = np.empty(PICTURE_SHAPE, np.float32)
            self.picture[0] = 1.0
            self.picture[1:] = np.asarray(PALETTE["sky"], np.float32)[:, None, None]
            self.enc_feats = np.zeros(self.encoder.out, np.float32)
        self.obs_dim = 61 + (self.encoder.out if self.encoder is not None else 0)
        self._trk = np.zeros(10)
        m = duck.model
        assert m.nq == info["nq"] and m.nv == info["nv"], (m.nq, m.nv, info["nq"], info["nv"])
        sp = info["spawn"]
        self.spawn = (float(sp["x"]), float(sp["y"]), float(sp.get("z") or 0.0), float(sp["yaw"]))
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, info.get("floor_geom", "floor"))
        root = m.body_rootid[m.body(info.get("duck_root", "trunk_base")).id]
        self.duck_mask = (m.body_rootid[m.geom_bodyid] == root)
        self.duck_root = int(root)
        fo = info.get("floor_out_before_s")
        self.out_before = float(cfg["out_before_s"]) if cfg.get("out_before_s") is not None else (float(fo) if fo is not None else None)
        # disc_turns (the saucer spike) stops counting at the first floor contact: tracked without a floor-out rule too.
        self.track_out = self.out_before is not None or info["metric"]["kind"] in ("disc_turns", "disc_peak_rpm")
        self.metric = make_metric(info, DT)
        self.ex = columns(info)
        self.dim = state_dim(info)
        self.adapter = make_adapter(self)
        # critic priv: the lab's 4 + [prop speed, t / duration] (6: the MLX learner's batch layout is fixed at 61 + 6);
        # an adapter may overwrite some of them with its own (priv_override)
        self.priv_dim = self.PRIV
        self.end_on_stop = bool(cfg.get("end_on_stop", False))
        self.end_on_finish = bool(cfg.get("end_on_finish", False))
        # cfg end_after_out_s (saucer targets E): end this long after the first floor contact's row (nothing later counts)
        eao = cfg.get("end_after_out_s")
        self.end_after_out_rows = None if eao is None else int(round(float(eao) / DT))
        self.prop_dof = int(info["props"][0]["dof_adr"]) if info.get("props") else None
        self.end_s = float(cfg.get("end_s") or self.adapter.default_end_s())
        self.episode_s = self.end_s
        self.ep_len = int(round(self.end_s / DT))
        self.fall_rule = cfg.get("fall", "tilt45")
        self.down_ticks = int(round(float(cfg.get("down_s", 3.0)) / DT))
        self.cos_max_tilt = math.cos(math.radians(cfg["max_tilt_deg"])) if cfg.get("max_tilt_deg") else None
        self.end_on_out = bool(cfg.get("end_on_out", True))
        self.term_at_end = bool(cfg.get("term_at_end", False))
        self.stagger = bool(cfg.get("stagger_start", False))
        self._first_train_reset, self._cut = True, False
        self.end_reason = None
        self.start = cfg.get("start", "spawn")
        self.bank = load_bank(cfg["bank"]) if self.start == "handoff" else []
        self.handoff_share = float(cfg.get("handoff_share", 1.0))
        self.act_rand = cfg.get("act_rand")
        # walk -> walk switches of the same policy: [(tick, node)] (the first node is cfg node, from tick 0)
        self.schedule = sorted((int(round(float(t) / DT)), nd) for t, nd in (cfg.get("schedule") or []))
        self.ctrl = self._ctrl_for(cfg["node"])
        # kit 1.15.0: a duck-program/2 trained node's feed and head holds, every program tick as TS (solo_program.py)
        self.sprog, self._head = None, None
        # (not in program mode, cfg program_nodes: there the whole duckscript runs, feed and head holds included)
        prog = cfg.get("program")
        if prog is not None and prog.get("format") == "duck-program/2" and cfg.get("program_nodes") is None:
            from .solo_program import SoloProgram
            sp = SoloProgram(info, prog, duck.model, duck, self.rng, cfg["node"].get("id") or prog["start"])
            self.sprog = sp if sp.active else None
        self._P0 = duck.bam.P.copy()
        self._vadr_legs = self._vadr[LEGS].copy()
        self._acc = np.zeros(3)
        jids = [int(np.nonzero(m.jnt_qposadr == a)[0][0]) for a in duck.qadr]
        self._jlo, self._jhi = m.jnt_range[jids, 0].copy(), m.jnt_range[jids, 1].copy()
        self.cmd = np.zeros(13)
        self.k = 0
        self.from_bank = False
        self.score = 0.0
        self._done_reset()
        # the fast rollout's fused substep loop (duck_rl/fused.py; bit-identical to the loop in step), None = stock
        self._fused = None
        if fused:
            from .fused import solo_tick
            self._fused = solo_tick(self)
        # the Challenge's runtime mechanics (duck_rl/movers.py; None: the old code path) and program mode (module doc)
        self.mech = make_mechanics(info, m)
        self.hit_t, self.hit_why, self._succeeded = None, None, False
        self.seed, self._n = int(seed), 0
        if self.mech is not None and self.start == "handoff":
            raise ValueError("a challenge with runtime mechanics (movers.py) trains from the spawn only (no handoff bank)")
        self.prog = None
        self.skipped = 0
        if cfg.get("program") is not None and cfg.get("program_nodes") is not None:
            self._program_setup()

    # ---- the host and the move's done rule ----
    def _ctrl_for(self, node: dict):
        """The command controller of a trained node: the raw host's block (raw_command[node id], with this episode's
        command_ranges draw), else the walk-stand controller of its skill and params."""
        if not self.host_raw:
            ctrl = make_controller(node)
            for i, v in self._draw.items():  # E2.1: walk-stand command draw (slots 0-2 only)
                if i == 0:
                    ctrl.vx = float(v); ctrl.moving = True
                elif i == 1:
                    ctrl.vy = float(v)
                elif i == 2:
                    ctrl.wz = float(v)
            if self._draw:
                ctrl.reset()
            return ctrl
        rc = self.cfg.get("raw_command") or {}
        nid = node.get("id")
        if nid not in rc:
            raise ValueError(f"host raw: no raw_command for node {nid!r} (raw_command has {sorted(rc)})")
        c = np.asarray(rc[nid], float).copy()
        for i, v in self._draw.items():
            c[i] = v
        return RawController(c)

    def _done_reset(self) -> None:
        """skills.ts SkillLayer.enter: the node's tick counter, the upright run and the tip flag start over."""
        self._node_k, self._up, self._tipped, self.move_done = 0, 0, False, False

    def _done_account(self, cos_tilt: float) -> None:
        """skills.ts #account on the state after a tick the move ran: gz = -cos_tilt; upright = tilt < 32 deg."""
        self._node_k += 1
        if cos_tilt < 0.0:
            self._tipped = True
        self._up = self._up + 1 if cos_tilt > COS_UPRIGHT else 0
        r = self.done_ticks
        if (("after_s" in r and self._node_k >= r["after_s"]) or ("upright_s" in r and self._up >= r["upright_s"])
                or ("rolled_then_upright_s" in r and self._tipped and self._up >= r["rolled_then_upright_s"])):
            self.move_done = True

    # ---- program mode (kit 1.9.0) ----
    def _program_setup(self) -> None:
        """The solo program runtime (module doc: program mode)."""
        from types import SimpleNamespace
        from .duel import load_runner
        from .program import EXTERNAL, HOST_SKILLS, SOLO_SIGNALS, CompiledProgram, DuckProgram, MatchRuntime, bind_moves
        from .movers import mover_signals
        cfg, info, m, duck = self.cfg, self.info, self.duck.model, self.duck
        if "runtime" not in info:
            raise ValueError(f"program mode: the info of {info['challenge']['id']} has no runtime block (ts/prepare.ts exports "
                             "it for a solo Challenge with a ported mechanic, duck_rl/movers.py)")
        bad = [k for k in ("schedule", "done") if cfg.get(k)] + (["start handoff"] if self.start == "handoff" else []) \
            + (["host raw"] if self.host_raw else [])
        if bad:
            raise ValueError(f"program mode: {', '.join(bad)} not supported (v1)")
        self.rt = rt = MatchRuntime(info)
        if rt.ticks_per_s != TICKS_PER_S or abs(rt.policy_dt - DT) > 1e-15:
            raise ValueError(f"program mode: runtime ticks {rt.ticks_per_s} / dt {rt.policy_dt} vs the env's {DT}")
        prog = CompiledProgram(cfg["program"], "program", (*SOLO_SIGNALS, *mover_signals(info)))
        if "progress" in prog.signals:
            raise ValueError("program mode: a `progress` guard is not ported (v1)")
        if info.get("pushes"):
            raise ValueError("program mode: a challenge with pushes is not ported (their draws come before the sensors' noise)")
        nodes = list(cfg.get("program_nodes") or [])
        for nid in nodes:
            if nid not in prog.nodes or prog.nodes[nid].skill not in HOST_SKILLS:
                raise ValueError(f"program mode: trained node {nid!r} is not a walk/stand node of the duckscript")
        stock = {}
        for name in sorted(set(rt.policy[s] for s in ("walk", "stand", "get_up"))):
            cn = duck.command_names(name)
            stock[name] = (load_runner(PLANT / "policies" / f"{name}.mlp.json"), "head_pose" in cn, "body_pose" in cn)
        base = [dict(e) for e in cfg.get("base_moves") or []]  # BASE_STRATEGY (kit 1.17.0): your earlier moves on the other nodes
        self.prog = DuckProgram(rt, prog, stock, bind_moves(prog, [dict(nodes=nodes, move=EXTERNAL), *base], rt, load_runner),
                                routes=info.get("routes"))
        self.prog_nodes = nodes
        ends = set(cfg.get("skip_end_skills", ("get_up",)))
        self.skip_end = {nid for nid, n in prog.nodes.items() if n.skill in ends}
        root = int(m.body_jntadr[duck.trunk])
        # program.py DuckProgram.step's skills layer also reads root_dof_adr, qadr and default_pos (kit 1.15.0); branch
        # kit 1.17.0 fix: DuckProgram.step reads the root's velocity (root_dof_adr) and the head joints' offset (qadr,
        # default_pos) on every tick after the first: they were missing, so a program-mode run stopped at tick 1
        self._pduck = SimpleNamespace(root_qpos_adr=int(m.jnt_qposadr[root]), root_dof_adr=int(m.jnt_dofadr[root]),
                                      trunk=duck.trunk, gyro_adr=duck.gyro_adr, qadr=duck.qadr, default_pos=duck.default_pos,
                                      camera_site=m.site(rt.raw["camera_site"]).id, ray_site=m.site(rt.raw["ray_site"]).id)
        self._allows_progress = "progress" in (info.get("signals_allowed") or ())
        self._act = np.zeros(14, np.float32)

    def _label(self, ep: dict | None) -> str:
        """The episode's PRNG label base (module doc: mechanics)."""
        challenge = self.info["challenge"]
        if ep is not None and ep.get("label"):
            return str(ep["label"])
        if ep is not None and ep.get("seed") is not None:
            return episode_label(rng_key(challenge), int(challenge["version"]), int(ep["seed"]))
        self._n += 1
        return f"{rng_key(challenge)}:{challenge['version']}:train:{self.seed}:{self._n - 1}"

    def _check_hit(self, t: float) -> None:
        """episode.ts: a mover's out rule on this row (data forwarded), before the finish only."""
        if self.hit_t is None and not self._succeeded:
            why = self.mech.out(self.duck.data, t)
            if why is not None:
                self.hit_t, self.hit_why = t, why

    # ---- resets ----
    def _spawn_pose(self, ep: dict) -> None:
        """episode.ts reset: q[0] = spawn.x + dx, q[1] = spawn.y + dy, q[2] += spawn.z, yaw = spawn.yaw + dyaw
        (lab.duck.apply_episode alone assumes spawn (0, 0, yaw 0))."""
        q = self.duck.data.qpos
        sx, sy, sz, syaw = self.spawn
        q[0], q[1] = sx + ep["dx"], sy + ep["dy"]
        q[2] += sz
        yaw = syaw + ep["dyaw"]
        q[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]

    def _node_at(self, k: int) -> dict:
        node = self.cfg["node"]
        for ks, nd in self.schedule:
            if ks <= k:
                node = nd
        return node

    def reset(self, ep: dict | None = None, cmd=None, bank_state: dict | None = None):
        """ep = explicit draws (dx, dy, dyaw, friction, mass) -> a Challenge reset (no noise, nominal props, spawn start);
        bank_state = a handoff state (tests); else a training draw."""
        train = ep is None and bank_state is None
        if os.environ.get("DUCK_CMD_DEBUG") and getattr(self, "_cmd_counts", None):
            print(f"[cmd] episode done switches={self._cmd_switches} resamples={self._cmd_resamples} "
                  f"dwell={self._cmd_dwell[-6:]} exposure={self._cmd_counts}", flush=True)
        duck = self.duck
        P = duck.bam.P
        P[:] = self._P0
        if train and self.act_rand:
            P[6] = self.rng.uniform(*ACT_RAND["vin"])
            P[0] *= self.rng.uniform(*ACT_RAND["kt"])
            P[1] *= self.rng.uniform(*ACT_RAND["R"])
        if bank_state is None and train and self.bank and self.rng.random() < self.handoff_share:
            bank_state = self.bank[int(self.rng.integers(len(self.bank)))]
        # command-conditioned training (raw host): one draw of the ranged slots per training episode
        self._draw = {i: float(self.rng.uniform(lo, hi)) for i, (lo, hi) in sorted(self._ranges.items())} if train else {}
        if train and self._draw and os.environ.get("DUCK_CMD_DEBUG"):
            print(f"[cmd] episode draw {self._draw}", flush=True)
        self._done_reset()
        self._staged = False
        if self.cmd_sched:
            sc = self.cmd_sched
            vals = list(sc.get("values") or [0.10, 0.19])
            self._cmd = float(self.rng.choice(vals)) if train else float(vals[-1])
            self._next_switch_t = self.rng.uniform(*sc.get("interval", [1.0, 3.0]))
            self._cmd_counts = {v: 0 for v in vals}
            self._cmd_resamples = 0
            self._cmd_switches = 0
            self._cmd_dwell = []
            self._cmd_last_t = 0.0
        else:
            self._cmd = None
        self.t_out = None
        self.score = 0.0
        self._down = 0
        d = duck.data
        if self.mech is not None:
            self.hit_t, self.hit_why, self._succeeded = None, None, False
            self.mech.reset(self._label(ep))
        if bank_state is not None:
            b = bank_state
            self.adapter.reset(train, spawn=False)
            self.ep = dict(b["draws"], pushes={})
            apply_episode(duck, self.ep, 0.0, self.rng)
            d.qpos[:] = b["qpos"]
            d.qvel[:] = b["qvel"]
            duck.prev_action = np.asarray(b["prev_action"], np.float32)
            duck.lag_joint_vel = np.asarray(b["lag_joint_vel"], np.float64)
            duck.head_cmd = np.asarray(b["head_ema"], np.float64).copy()
            tgt = duck.default_pos + duck.prev_action.astype(np.float64)
            duck.bam.ring[:] = tgt
            duck.bam.targets[:] = tgt
            self.k = int(b["tick"])
            self.ctrl = self._ctrl_for(self._node_at(self.k))
            self.ctrl.reset()
            if b.get("was_moving"):
                self.ctrl.kick = 0  # a walk -> walk switch keeps the gait: no new start kick (skills.ts enter)
            self.metric.restore(b.get("metric_tail"), self.k)
            self.from_bank = True
        else:
            self.adapter.reset(train, spawn=True)
            self.ep = dict(ep, pushes={}) if ep is not None else draw_episode(self.rng, self.end_s, PUSH_OFF)
            apply_episode(duck, self.ep, self.noise if train else 0.0, self.rng)
            self._spawn_pose(self.ep)
            self.adapter.spin(train)
            if self.mech is not None:
                self.mech.apply(d, 0.0)  # episode.ts: drives.apply(data, 0) inside the reset
            self.k = 0
            self.ctrl = self._ctrl_for(self.cfg["node"])
            self.ctrl.reset()
            self.metric.reset()
            self.from_bank = False
        self.ep_len = int(round(self.end_s / DT))
        self._cut = False
        if train and self._first_train_reset:
            self._first_train_reset = False
            if self.stagger and self.ep_len - self.k > 1:  # STAGGER_START: the first episode ends at a random tick
                cut = self.k + int(self.rng.integers(1, self.ep_len - self.k + 1))  # (drawn last: the reset's own
                self._cut, self.ep_len = cut < self.ep_len, cut  # draws are those of an unstaggered run)
        self._fresh = True  # the node-entry tick: no controller observe (TS observes before enter)
        self._head = None
        if self.sprog is not None:
            self.sprog.reset(self.k)
        tr = self.tracker = Tracker(duck, self.end_s)
        self._trk[:] = [tr.x0, tr.y0, tr.yaw0, tr.yaw_prev, tr.yaw_cum, 0.0, 0.0, 0.0, tr.prev["height"], 0.0]
        if self.prog is not None:
            from .program import SoloSensors
            lab_ = self.mech.label if self.mech is not None else self._label(ep)
            prng = label_prng(lab_)  # episode.ts: the episode stream, past drawRandomisation (no pushes), feeds the sensors' noise
            draw_randomisation(prng, self.rt.raw["dr"])
            sens = SoloSensors(duck.model, self.rt, self.prog.program, prng, self._pduck, self.mech, self._allows_progress,
                               target=self.info.get("target"), goal=self.info.get("goal"))
            self.prog.reset(duck.model, self._pduck, sens.prng, -1, None, 0.0, sensors=sens)
            self.skipped = 0
        # a handoff starts at any tick: the camera's picture is refreshed here, not left from
        # the last episode until the next program tick (_obs refreshes every VISION_EVERY rows)
        self._view_stale = True
        obs = self._obs()
        if self.mech is not None:
            self._check_hit(0.0)
        # row k (the reset state), as TS updates the metric on every row
        self.metric.push(d.qpos, d.qvel, out=False, cos_tilt=float(d.xmat[duck.trunk][8]))
        self._metric_prev = self._metric_now()
        # kit 1.17.0: the duckscript may start on (or pass straight to) a node the policy does not run (a stock skill, or a
        # BASE_STRATEGY move): those ticks run here, as step() runs them between trained ticks, until a trained node runs
        while self.prog is not None and not self.prog.external:
            self.prog.runner.run(obs[0], self._act)  # obs = (obs, priv), as _obs() returns it
            o1, p1, _, term, trunc = self._step1(self._act.copy())
            obs = (o1, p1)
            if term or trunc:
                raise RuntimeError(f"program mode: the episode ended before any trained node ran (TRAINED_NODE "
                                   f"{self.prog_nodes}; the duckscript was on {self.prog.node_name!r})")
        if self.prog is not None:
            self._metric_prev = self._metric_now()
        return obs

    def _metric_now(self) -> float:
        lost = (self.t_out is not None and self.out_before is not None and self.t_out < self.out_before) \
            or self.hit_t is not None
        return 0.0 if lost else self.metric.value

    def _obs(self):
        duck = self.duck
        d = duck.data
        mujoco.mj_forward(duck.model, d)
        if self.prog is not None:  # program mode: the duckscript's row k (posture, skills layer, program tick, command)
            p = self.prog
            p.observe_row(float(d.xmat[duck.trunk][8]))
            p.step(self.k, d)
            self.cmd = p.command.copy()
            self._head = p.head  # duck-program/2 head holds of the running node (None: the move's own)
            use_head, use_body = p.head_in_obs, p.body_in_obs
        else:
            if not self._fresh:
                self.ctrl.observe(float(d.sensordata[duck.gyro_adr + 2]))
            self._fresh = False
            sp = self.sprog
            if sp is not None:  # episode.ts: the row's fall counter, then (every PROGRAM_EVERY rows) the program tick
                sp.observe_row(float(d.xmat[duck.trunk][8]))
                if self.k % sp.rt.program_every == 0:
                    feed, head = sp.tick(self.k, d, bool(self.move_done))
                    if sp.node.feed:
                        self.ctrl.feed(feed, sp.rt.walk_ranges)
                    if sp.node.head:
                        self._head = head
            if self.cmd_sched and not self.host_raw:
                sc = self.cmd_sched
                vals = list(sc.get("values") or [0.10, 0.19])
                t = self.k * 0.02
                if sc.get("mode") == "time" and self.k % 50 == 25:  # every 1 s of policy time (50 Hz: 50 ticks)
                    if t >= self._next_switch_t:
                        new = float(self.rng.choice(vals))
                        self._cmd_resamples += 1
                        if new != self._cmd:
                            self._cmd_switches += 1
                            self._cmd_dwell.append(round(t - self._cmd_last_t, 2))
                            self._cmd_last_t = t
                            self._cmd = new
                        self._next_switch_t = t + self.rng.uniform(*sc.get("interval", [1.0, 3.0]))
                elif sc.get("mode") == "state":
                    h = float(d.qpos[2])
                    want = vals[-1] if h >= float(sc.get("height", 0.15)) else vals[0]
                    if want != self._cmd:
                        self._cmd = want
                        self._cmd_switches += 1
                if self._cmd is not None:
                    self.ctrl.vx = self._cmd; self.ctrl.moving = True
                    if self._cmd in self._cmd_counts:
                        self._cmd_counts[self._cmd] += 1
            if (self.cmd_stage and not self.host_raw and not self._staged
                    and float(d.qpos[2]) >= float(self.cmd_stage["height"])):
                self.ctrl.vx = float(self.cmd_stage["vx"]); self.ctrl.moving = True; self._staged = True
                if os.environ.get("DUCK_CMD_DEBUG"):
                    print(f"[cmd] staged vx={self.cmd_stage['vx']} at z={float(d.qpos[2]):.3f}", flush=True)
            self.cmd = self.ctrl.command()
            if os.environ.get("DUCK_CMD_DEBUG") and self.cmd_sched and self.k == 100:
                print(f"[cmd] obs twist at k=100: {np.round(self.cmd[:3], 4).tolist()} (walk cmd vx={self.ctrl.vx})", flush=True)
            use_head, use_body = self._use_head, self._use_body
        lag = duck.lag_joint_vel
        nb_obs(self._ob, self._pr, self._vn, duck.head_cmd, d.sensordata, duck.gyro_adr, d.xquat[duck.trunk], d.qpos,
               d.qvel, self._qadr, self._vadr, duck.default_pos, lag if lag is not None else self._vn, lag is not None,
               duck.prev_action, self.cmd, use_head, use_body, self.ep["friction"], self.ep["mass"])
        self._vel_now = self._vn.copy()
        self._pr[4] = d.qvel[self.prop_dof] if self.prop_dof is not None else 0.0
        self._pr[5] = self.k * DT / self.challenge_s
        if hasattr(self.adapter, "priv_override"):
            self.adapter.priv_override(d, self._pr)
        refresh = self.k % VISION_EVERY == 0 or self._view_stale
        self._view_stale = False
        if self.cam is not None:
            if refresh:
                self.cam.refresh(d, self.picture)
                if self.enc_defer:
                    self.enc_dirty = True
                else:
                    self.enc_feats = self.encoder.features(self.picture)
            return np.concatenate([self._ob, self.enc_feats]), self._pr.copy()
        return self._ob.copy(), self._pr.copy()

    def step(self, action):
        if self.prog is None:
            return self._step1(action)
        E = self.ex
        m0 = self._metric_prev
        out = self._step1(action)  # program mode: the policy's tick, then untrained nodes run stock (module doc)
        self.skipped = 0
        fin = float(out[2][E["finish_now"]]) if "finish_now" in E else 0.0
        while not (out[3] or out[4]) and not self.prog.external:
            node = self.prog.node
            if node is not None and node.id in self.skip_end:  # a fall: the duckscript gets up; the episode ends here
                out[2][FELL_I] = 1.0
                out[2][TERM_I] = 1.0
                self.end_reason = "fall"
                out = (out[0], out[1], out[2], True, False)
                break
            self.prog.runner.run(out[0], self._act)
            out = self._step1(self._act.copy())
            self.skipped += 1
            if "finish_now" in E:
                fin = max(fin, float(out[2][E["finish_now"]]))
        st = out[2]
        st[E["ticks"]] = 1 + self.skipped
        st[E["dmetric"]] = st[E["metric"]] - m0
        if "finish_now" in E:
            st[E["finish_now"]] = fin
        return out

    def _step1(self, action):
        """One 50-Hz tick with `action` and its row (the whole step outside program mode)."""
        duck = self.duck
        action = np.asarray(action, np.float32)
        prev = duck.prev_action.copy()
        if self._head is not None:  # sim.ts holdHead: the held head joints over the policy's action
            from .solo_program import hold_head
            action = action.copy()
            hold_head(action, prev, self._head, (self.sprog.rt if self.sprog is not None else self.rt).head_ema)
        bam, m, dd = duck.bam, duck.model, duck.data
        bam.latch(duck.default_pos + 1.0 * action.astype(np.float64))
        acc = self._acc
        acc[:] = 0.0
        tlim = bam.P[0] * bam.P[5] * 0.999
        t_tick = self.k * DT
        check_out = self.track_out
        if self.mech is not None:
            self.mech.apply(dd, (self.k + 1) / TICKS_PER_S)  # episode.ts: drives right before tick k
        s_f = None  # the fused kernel: the same substeps in one nogil call (None: the stock loop below)
        if self._fused is not None and self._fused.ready(m, dd):
            s_f = self._fused.run(bam, acc, tlim, bool(check_out and self.t_out is None))
        if s_f is not None:
            if s_f >= 0:
                self.t_out = t_tick + s_f * DT / duck.decimation
        else:
            for s in range(duck.decimation):
                bam.before_substep(dd)
                _effort(acc, bam.prev_torque, dd.qvel, LEGS, self._vadr_legs, tlim)
                mujoco.mj_step(m, dd)
                if check_out and self.t_out is None and dd.ncon and _floor_touch(dd.contact.geom, dd.ncon, self.floor, self.duck_mask):
                    self.t_out = t_tick + s * DT / duck.decimation  # the contacts of the state this substep started from
        nsub = duck.decimation
        cu, mech, lim = acc
        duck.prev_action = action
        duck.lag_joint_vel = self._vel_now
        self.k += 1
        switched = False
        for k_sw, nd in self.schedule:
            if self.k == k_sw:
                self.ctrl = self._ctrl_for(nd)
                self.ctrl.kick = 0  # a walk -> walk switch while moving: no new start kick
                self._fresh = True  # TS observes on the old layer, then enters the new one
                switched = True
        obs, priv = self._obs()
        d = duck.data
        if self.mech is not None:
            self._check_hit(self.k * DT)
        if switched:
            self._done_reset()  # the new node has run no tick yet
        elif self.done_ticks:
            self._done_account(float(d.xmat[duck.trunk][8]))
        t = self.k * DT
        if check_out and self.t_out is None and d.ncon and _floor_touch(d.contact.geom, d.ncon, self.floor, self.duck_mask):
            self.t_out = t
        self.metric.push(d.qpos, d.qvel, out=self.t_out is not None, cos_tilt=float(d.xmat[duck.trunk][8]))
        lost = self.t_out is not None and self.out_before is not None and self.t_out < self.out_before
        if self.hit_t is not None:
            lost = True
        mval = self._metric_now()
        self.score = mval
        settled = self.metric.settled
        xm = d.xmat[duck.trunk]
        upright = xm[8] > COS_UPRIGHT
        if self.fall_rule == "tilt45":
            fell = is_fall(duck)
        elif self.fall_rule == "down":
            self._down = 0 if upright else self._down + 1
            fell = self._down > self.down_ticks
        else:
            fell = False
        if self.cos_max_tilt is not None and xm[8] < self.cos_max_tilt:
            fell = True
        ended = bool(self.end_on_out and lost)
        why = "out" if ended else None
        if self.end_on_stop and getattr(self.metric, "stopped", False):
            ended = True  # cfg end_on_stop: the metric froze (disc_turns: floor contact or on-disc fall), nothing more counts
            why = why or "stop"
        if self.end_after_out_rows is not None and self.t_out is not None \
                and self.k >= math.ceil(self.t_out / DT - 1e-9) + self.end_after_out_rows:
            ended = True
            why = why or "stop"
        if self.end_on_done and self.move_done:
            ended = True  # cfg end_on_done: the move's done rule fired; the duckscript's next node takes over
            why = "done" if why in (None, "stop") else why
        success = settled and not math.isnan(mval) and meets_success(self.info, mval)
        if self.end_on_finish and success:
            ended = True  # cfg end_on_finish: the race is finished (the TS `success` event); nothing later counts
            why = "finish"
        term = fell or ended
        st = np.empty(self.dim, np.float32)
        nb_state(st, self._trk, xm, d.qpos, d.qvel, d.sensordata, duck.gyro_adr, d.xpos[duck.head_body, 2],
                 d.geom_xpos, self._soles, d.contact.geom, d.ncon, duck.bam.prev_torque, self._qadr, self._vadr,
                 duck.default_pos, action, prev, self.cmd, self.k, DT, self.challenge_s, float(fell), float(term), 0.0)
        E = self.ex
        st[E["x"]], st[E["y"]], st[E["hdg"]] = d.qpos[0], d.qpos[1], heading(xm)
        st[E["up32"]], st[E["ended"]] = float(upright), float(ended)
        st[E["out"]], st[E["lost"]] = float(self.t_out is not None), float(lost)
        st[E["metric"]], st[E["dmetric"]] = mval, mval - self._metric_prev
        st[E["settled"]] = float(settled)
        st[E["move_done"]] = float(self.move_done)
        st[E["success"]] = float(success)
        self._metric_prev = mval
        for n, v in self.metric.columns().items():
            st[E[n]] = v
        if "ticks" in E:
            st[E["ticks"]] = 1.0
        if self.mech is not None:
            self._succeeded = self._succeeded or bool(success)
            for mv in self.mech.movers:  # the out column of the mover whose rule fired (Traffic: hit, Light: red_out)
                st[E[mv.COLUMNS[0]]] = float(self.hit_why == mv.WHY)
            if self.hit_t is not None and "finish_t" in E:  # an out = never finishes (the TS metric is 0 from then on)
                st[E["finished"]], st[E["finish_now"]], st[E["finish_t"]] = 0.0, 0.0, self.metric.MISS_S
        if "score" in E:  # disc_speed: the Challenge score from the bell row on
            st[E["score"]] = mval if settled else 0.0
        self.adapter.fill(st, E, d)
        P = bam.P
        st[E["copper"]] = cu / nsub / (P[0] * P[0]) * P[1]
        st[E["mech"]], st[E["atlim"]] = mech / nsub, lim / nsub
        tgt = duck.default_pos[LEGS] + action[LEGS]
        st[E["oor"]] = float(np.sum(np.maximum(0.0, tgt - self._jhi[LEGS]) + np.maximum(0.0, self._jlo[LEGS] - tgt)))
        timeout = (not term) and self.k >= self.ep_len
        if timeout and self.term_at_end and not self._cut:  # (a STAGGER_START cut stays a bootstrapped time-out)
            term, timeout = True, False
        self.end_reason = ("fall" if fell else why or "timeout") if term or timeout else None
        return obs, priv, st, term, timeout

    def episode_stats(self, s: np.ndarray, term: bool) -> dict:
        """The pool's record of the episode that just ended on state row s (train.py's per-iteration log)."""
        ex = self.ex
        return dict(len=self.k, fell=float(s[FELL_I] > 0), terminated=float(term), disp=self.disp_now,
                    lost=float(s[ex["lost"]] > 0), score=float(self.score), success=float(s[ex["success"]] > 0),
                    t_end=self.k * DT, bank=float(self.from_bank), reason=self.end_reason)


def make_env(cfg: dict, seed: int, fused: bool = False):
    """The training env of a cfg: DuelEnv for a duel Challenge (info `duel`: duck_rl/duel_env.py), else ChallengeEnv. fused: the
    fast rollout's fused substep loop (duck_rl/fused.py; the stock loop where it is not available)."""
    if "duel" in cfg["info"]:
        from .duel_env import DuelEnv
        return DuelEnv(cfg, seed, fused=fused)
    return ChallengeEnv(cfg, seed, fused=fused)


def bigbuf(conn, size: int = 4 << 20) -> bool:
    """4 MB socket buffers on a multiprocessing Pipe end (the fast rollout). macOS' AF_UNIX default is 8 KB, so a worker's
    15-30 KB reply blocks it until the trainer reads it: with the pipelined rollout a worker would stall for the other
    half's policy forward every step (and could block for good on its reply if the trainer stops reading mid-step).
    -> True when both buffers now hold >= size (read back), False when they could not be enlarged (a Pipe that is not a
    socket, a capped buffer): ChallengePool then runs lock-step."""
    import socket
    try:
        sk = socket.socket(fileno=os.dup(conn.fileno()))  # (a dup of the same socket: closing it keeps the Pipe's fd)
        try:
            for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
                sk.setsockopt(socket.SOL_SOCKET, opt, size)
            return all(sk.getsockopt(socket.SOL_SOCKET, opt) >= size for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF))
        finally:
            sk.close()
    except (OSError, ValueError):
        return False


def _fused_on(e) -> bool:
    """Whether env e runs the fused substeps (verified: duck_rl/fused.py), a ChallengeEnv's or a DuelEnv's world's."""
    k = getattr(getattr(e, "world", e), "_fused", None)
    return k is not None and k.ok is True


def flush_encoders(envs: list, obs: np.ndarray | None = None) -> None:
    """The deferred encoder forwards (ChallengeEnv enc_defer, cfg "encoder"): one batched forward over the envs whose
    picture refreshed since the last flush; their held features, and their rows of `obs` (row i = envs[i]), take the
    result. The per-env forward's numbers to float32 rounding (the batched matrix products sum in another order: ~1e-6)."""
    dirty = [i for i, e in enumerate(envs) if getattr(e, "enc_dirty", False)]
    if not dirty:
        return
    F = envs[dirty[0]].encoder.features(np.stack([envs[i].picture for i in dirty]))
    for j, i in enumerate(dirty):
        e = envs[i]
        e.enc_feats, e.enc_dirty = F[j], False
        if obs is not None:
            obs[i, e.obs_dim - e.encoder.out:e.obs_dim] = F[j]


# one thread per worker for every BLAS numpy may use (OpenMP, OpenBLAS, MKL, macOS Accelerate): the encoder's batched
# matrix products would otherwise start a thread pool in each worker and oversubscribe the cores
ONE_THREAD = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}


def _worker(remote, cfg, seeds, w=0, fused=False):
    import signal
    import time
    for k, v in ONE_THREAD.items():
        os.environ.setdefault(k, v)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # the budget runner's SIGTERM is for the trainer (it saves, then closes us)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    worlds = cfg.get("worlds")  # per-env worlds (train.py WORLDS): env i of the pool gets worlds[i % n]
    envs = [make_env(dict(cfg, scene=worlds[(w * len(seeds) + i) % len(worlds)]["scene"],
                          info=worlds[(w * len(seeds) + i) % len(worlds)]["info"]) if worlds else cfg, s, fused)
            for i, s in enumerate(seeds)]
    for e in envs:  # a frozen camera encoder: one batched forward per step over the envs that refreshed (flush_encoders)
        e.enc_defer = getattr(e, "encoder", None) is not None
    first = [e.reset() for e in envs]
    flush_encoders(envs)
    for (o, _), e in zip(first, envs):
        if e.enc_defer:
            o[e.obs_dim - e.encoder.out:] = e.enc_feats
    dim, hk = envs[0].dim, len(envs) // 2

    def run(sub, data):
        t0 = time.perf_counter()
        n = len(sub)
        obs = np.empty((n, sub[0].obs_dim), np.float32)  # 61, or 61 + the encoder's out
        priv = np.empty((n, sub[0].priv_dim), np.float32)
        S, term, tout, eps = np.empty((n, dim), np.float32), np.zeros(n, bool), np.zeros(n, bool), []
        for i, (e, a) in enumerate(zip(sub, data)):
            o, p, s, tm, to = e.step(a)
            S[i], term[i], tout[i] = s, tm, to
            if tm or to:
                eps.append(e.episode_stats(s, tm))
                o, p = e.reset()
            obs[i], priv[i] = o, p
        flush_encoders(sub, obs)
        # (8th: this worker's busy seconds for the step: ChallengePool's straggler stats; 9th: its envs on fused substeps)
        return (obs, priv, S, term, tout, eps, np.array([e.k for e in sub]), time.perf_counter() - t0,
                sum(_fused_on(e) for e in envs))

    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            remote.send((np.stack([o for o, _ in first]), np.stack([p for _, p in first])))
        elif cmd == "step":  # all of this worker's envs
            remote.send(run(envs, data))
        elif cmd == "steph":  # one half (the pipelined rollout): (h, the actions of envs[:hk] (h 0) or envs[hk:] (h 1))
            h, act = data
            remote.send(run(envs[:hk] if h == 0 else envs[hk:], act))
        elif cmd == "close":
            remote.close()
            return


class ChallengePool(Pool):
    """lab.pool.Pool with ChallengeEnv workers (DuelEnv ones for a duel Challenge: make_env). The fast rollout (kit 1.8.0,
    duck_rl/rollout.py): fused = the envs' fused substep loop (duck_rl/fused.py); pipe = each worker's envs step as two
    halves, so the trainer acts on one half while the workers step the other (duck_rl/rollout.py Pipeliner drives it;
    without one the pool is lock-step, as stock); big = 4 MB pipe buffers. Every option is bit-identical to stock."""

    def __init__(self, cfg: dict, workers: int = 8, per_worker: int = 16, seed: int = 0, fused: bool = False,
                 pipe: bool = False, big: bool = False):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        big_ok = bool(big)
        for w in range(workers):
            a, b = ctx.Pipe()
            if big:  # both ends, before the worker gets b (it receives the same socket)
                big_ok = bigbuf(a) and bigbuf(b) and big_ok
            seeds = [seed * 100_000 + w * 1000 + i for i in range(per_worker)]
            p = ctx.Process(target=_worker, args=(b, cfg, seeds, w, fused), daemon=True)
            # a spawned worker loads numpy (and its BLAS, which reads these once) before _worker runs: it gets them in
            # the environment it starts with; the trainer's own environment is left as it was
            saved = {k: os.environ.get(k) for k in ONE_THREAD}
            os.environ.update({k: os.environ.get(k) or v for k, v in ONE_THREAD.items()})
            try:
                p.start()
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            self.remotes.append(a)
            self.procs.append(p)
        self.per, self.n = per_worker, workers * per_worker
        self.fused, self.big, self.big_ok = fused, big, big_ok
        self.fused_envs = None  # envs on fused substeps, as the workers last reported (None before the first step)
        # the pipelined rollout's halves: half 0 = each worker's first per_worker // 2 envs, half 1 = the rest;
        # hs[h] = the half's envs per worker, idx[h] = its global rows (worker by worker, env by env). Only with the
        # big buffers: with 8 KB ones a worker blocks on its reply while the trainer acts on the other half
        self.pipe = bool(pipe) and per_worker >= 2 and big_ok
        self.pipe_off = None if self.pipe or not pipe else (
            "1 env per worker" if per_worker < 2 else "the 4 MB pipe buffers could not be set")
        if self.pipe_off:
            print(f"[duck-rl] fast rollout: no pipelined halves ({self.pipe_off}): the pool runs lock-step", flush=True)
        hk = per_worker // 2
        self.hs = (hk, per_worker - hk)
        self.idx = [np.concatenate([w * per_worker + np.arange(lo, hi) for w in range(workers)])
                    for lo, hi in ((0, hk), (hk, per_worker))]
        self.hook, self.sent = None, [False, False]
        self.reset_timing()

    def close(self) -> None:
        """Close the workers; any still alive after the join (e.g. blocked sending a reply nobody reads after an exception
        mid-step; they ignore SIGTERM) are killed, so a failed run never hangs on shutdown."""
        self.sent = [False, False]
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=2)
        for p in self.procs:
            if p.is_alive():
                p.kill()
                p.join(timeout=2)
        for r in self.remotes:
            with contextlib.suppress(OSError):
                r.close()

    def reset_timing(self) -> None:
        """Per-worker busy seconds since the last call (train.py: per iteration) and the lock-step cost: the sum over
        steps of the slowest worker's time vs of the mean worker's."""
        self.busy = np.zeros(len(self.remotes))
        self.step_max = self.step_mean = 0.0

    def send_half(self, h: int, act_h: np.ndarray) -> None:
        """The pipelined rollout: send half h its actions (rows in idx[h] order); step() collects both halves."""
        assert self.pipe and not self.sent[h]
        k = self.hs[h]
        for w, r in enumerate(self.remotes):
            r.send(("steph", (h, act_h[w * k:(w + 1) * k])))
        self.sent[h] = True

    def step(self, actions: np.ndarray):
        """One tick of every env. Lock-step (stock) unless both halves were sent (send_half): then actions is ignored,
        and as soon as a half's results arrive, hook(h, obs_h, priv_h) may act on them and send that half's next
        actions (the Pipeliner) while the workers still step the other half."""
        if not (self.sent[0] or self.sent[1]):
            for i, r in enumerate(self.remotes):
                r.send(("step", actions[i * self.per:(i + 1) * self.per]))
            res = [r.recv() for r in self.remotes]
            self.fused_envs = sum(x[8] for x in res)
            t = np.array([x[7] for x in res])
            self.busy += t
            self.step_max += float(t.max())
            self.step_mean += float(t.mean())
            cat = lambda k: np.concatenate([x[k] for x in res])  # noqa: E731
            return cat(0), cat(1), cat(2), cat(3), cat(4), [e for x in res for e in x[5]], cat(6)
        assert self.sent[0] and self.sent[1], "send_half(0) and send_half(1) before step()"
        self.sent = [False, False]
        ra = [r.recv() for r in self.remotes]
        if self.hook is not None:  # act on half A and send its next actions while the workers step half B
            self.hook(0, np.concatenate([x[0] for x in ra]), np.concatenate([x[1] for x in ra]))
        rb = [r.recv() for r in self.remotes]
        if self.hook is not None:  # and half B's, before the trainer's reward and storage of this step
            self.hook(1, np.concatenate([x[0] for x in rb]), np.concatenate([x[1] for x in rb]))
        self.fused_envs = sum(x[8] for x in rb)
        t = np.array([xa[7] + xb[7] for xa, xb in zip(ra, rb)])
        self.busy += t
        self.step_max += float(t.max())
        self.step_mean += float(t.mean())
        ia, ib = self.idx

        def cat(k):
            a, b = np.concatenate([x[k] for x in ra]), np.concatenate([x[k] for x in rb])
            out = np.empty((self.n,) + a.shape[1:], a.dtype)
            out[ia], out[ib] = a, b
            return out
        eps = [e for xa, xb in zip(ra, rb) for e in xa[5] + xb[5]]  # stock's order: worker by worker, env by env
        return cat(0), cat(1), cat(2), cat(3), cat(4), eps, cat(6)

    def timing(self) -> dict:
        """The straggler stats since reset_timing(): wk_t_max / wk_t_min = the busiest / idlest worker's seconds,
        wk_ratio = their ratio, wk_lockstep = sum of per-step maxima / sum of per-step means (1 = no straggler)."""
        if not self.busy.any():
            return {}
        mx_, mn_ = float(self.busy.max()), float(self.busy.min())
        return dict(wk_t_max=round(mx_, 3), wk_t_min=round(mn_, 3), wk_ratio=round(mx_ / mn_, 3) if mn_ > 0 else None,
                    wk_lockstep=round(self.step_max / self.step_mean, 3) if self.step_mean > 0 else None)
