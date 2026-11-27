"""The fixed plant: plant v1 (tools/py/duckref DuckRefBam) + a numba BAM + Dare-like randomisation + pushes + duck state.

Nothing here is editable by move authors. Sequencing per policy tick k mirrors DuckRefBam.tick (plant-v0 §5):
mj_forward -> obs_k (61-D, runtime build_obs) -> a_k -> targets = default + 1.0 a_k -> 4 x (BAM, mj_step).
Randomisation mirrors packages/sim/src/runtime/episode.ts drawRandomisation + drawPushes (see README).
"""
from __future__ import annotations

import copy
import math

import mujoco
import numpy as np
from numba import njit

from . import PLANT  # noqa: F401  (sets sys.path for duckref)
from duckref.bam import FRICTION_DOF, BamActuator, DuckRefBam

DT = 0.02
FALL_TILT_DEG = 45.0  # fixed fall termination (smoke env), plus trunk z < FALL_Z
FALL_Z = 0.07
STAND_Z = 0.12
SLOTS = ["vx", "vy", "wz", "neck_pitch", "head_pitch", "head_yaw", "head_roll",
         "body_x", "body_y", "body_z", "body_roll", "body_pitch", "body_yaw"]

# ---- Dare randomisation (packages/sim/src/runtime/constants.ts DR, drawn in this order) + pushes -----------------
DR = dict(spawn_xy=0.03, spawn_yaw=math.radians(5.0), sole_friction=(0.7, 1.3), trunk_mass=(0.95, 1.05))
# Dare `pushes` mechanism (episode.ts drawPushes): first_s, gap U(every_s), speed U(speed), direction U(-pi, pi);
# dv is added to the trunk free-joint xy velocity before the policy tick. Parameters: microduck_rl's training
# push envelope (interval 3-6 s, |dv| <= ~0.3-0.4 m/s), not Hold the spot's 0.4-0.7 m/s shoves.
PUSH_TRAIN = dict(first_s=(1.0, 4.0), every_s=(3.0, 6.0), speed=(0.1, 0.35))
PUSH_PROBE = dict(first_s=(4.0, 4.0), every_s=(3.0, 5.0), speed=(0.1, 0.35))
RESET_JOINT_NOISE = 0.03  # training only (probes use the exact Dare reset: default pose)


# ---- numba BAM: the same equations as duckref BamActuator.before_substep, sums in joint order ------------------------
@njit(cache=True)
def _bam_substep(ring, head, delay, targets, qpos, qvel, qadr, dof, prev_torque, P, efc_type, efc_id, efc_force,
                 nefc, fric_type, qfrc_bias, qfrc_constraint, qfrc_actuator, frictionloss, ctrl, act, nv):
    size = ring.shape[0]
    head = (head + 1) % size
    n = targets.shape[0]
    for j in range(n):
        ring[head, j] = targets[j]
    lag = (head - delay) % size
    kt, R, kp, eg, max_pwm, max_cur = P[0], P[1], P[2], P[3], P[4], P[5]
    vin0, gain, vmin, fscale = P[6], P[7], P[8], P[9]
    load = 0.0
    for j in range(n):
        load += abs(prev_torque[j])
    vin = max(vin0 - gain * load, vmin)
    fric = np.zeros(nv)
    for r in range(nefc):
        if efc_type[r] == fric_type:
            fric[efc_id[r]] += efc_force[r]
    for j in range(n):
        q = qpos[qadr[j]]
        dq = qvel[dof[j]]
        duty = (ring[lag, j] - q) * kp * eg
        center = kt * dq / vin
        span = R * max_cur / vin
        duty = min(max(duty, center - span), center + span)
        duty = min(max(duty, -max_pwm), max_pwm)
        tau = kt * (vin * duty) / R - (kt * kt) * dq / R
        t_ext = -qfrc_bias[dof[j]] + qfrc_constraint[dof[j]] - fric[dof[j]]
        t_m = qfrc_actuator[dof[j]]
        s = math.exp(-math.pow(abs(dq) / P[18], P[19]))
        fl = P[10] + s * P[11]
        fl = fl + abs(t_ext * P[12] - t_m * P[13])
        fl = fl + s * abs(t_ext * P[14] - t_m * P[15])
        a_ext, a_mot = abs(t_ext), abs(t_m)
        quad = P[16] * (a_ext * a_ext) if a_mot > a_ext else P[17] * (a_mot * a_mot)
        fl = fl + s * quad
        frictionloss[dof[j]] = fl * fscale
        ctrl[act[j]] = tau
        prev_torque[j] = tau
    return head


class NumbaBam(BamActuator):
    def __init__(self, m, qadr, dof, act, cfg):
        super().__init__(m, qadr, dof, act, cfg)
        p = cfg.p
        self.P = np.array([p["kt"], p["R"], cfg.kp_fw, cfg.error_gain, cfg.max_pwm, cfg.max_current, cfg.vin,
                           cfg.vin_drop_gain, cfg.vin_min, cfg.friction_scale, p["friction_base"], p["friction_stribeck"],
                           p["load_friction_external"], p["load_friction_motor"], p["load_friction_external_stribeck"],
                           p["load_friction_motor_stribeck"], p["load_friction_external_quad"],
                           p["load_friction_motor_quad"], p["dtheta_stribeck"], p["alpha"]], np.float64)
        self.qadr_i, self.dof_i, self.act_i = (np.asarray(x, np.int64) for x in (qadr, dof, act))
        self._views = None

    def before_substep(self, d):
        if self._views is None:
            self._views = (d.qpos, d.qvel, d.qfrc_bias, d.qfrc_constraint, d.qfrc_actuator, self.m.dof_frictionloss, d.ctrl)
        qpos, qvel, qb, qc, qa, fl, ctrl = self._views
        self.head = _bam_substep(self.ring, self.head, self.cfg.delay_substeps, self.targets, qpos, qvel, self.qadr_i,
                                 self.dof_i, self.prev_torque, self.P, d.efc_type, d.efc_id, d.efc_force, d.nefc,
                                 FRICTION_DOF, qb, qc, qa, fl, ctrl, self.act_i, self.m.nv)


_PLANT_MODEL = None


def plant_model() -> mujoco.MjModel:
    """A fresh copy of the compiled plant v1 model: the XML is parsed once per process (21 ms) and copied per duck
    (3.5 ms, M: mac-speed/cpu/RESULTS.md). Each duck needs its own copy, because the BAM writes frictionloss and the
    episode randomisation writes friction and mass into the model. Same trajectories as a fresh parse, bit for bit
    (lab/selftest.py model-copy)."""
    global _PLANT_MODEL
    if _PLANT_MODEL is None:
        from duckref.bam import V1_BUNDLE_DIR
        from duckref.plant import load_manifest, load_model
        mjcf = next(f["path"] for f in load_manifest(V1_BUNDLE_DIR)["files"] if f["role"] == "mjcf")
        _PLANT_MODEL = load_model(V1_BUNDLE_DIR / mjcf)  # physics only when the bundle has no meshes/ (a player's kit)
    return copy.copy(_PLANT_MODEL)


def make_duck(bam: str = "numba", model: mujoco.MjModel | None = None) -> DuckRefBam:
    """bam = 'numba' (training env) or 'ref' (duckref's reference BAM, the replay/probe path). model: a compiled plant
    model the duck owns (plant_model()); default: parse the XML."""
    duck = DuckRefBam(model=model)
    if bam == "numba":
        duck.bam = NumbaBam(duck.model, duck.qadr, duck.vadr, duck.act, duck.bam_cfg)
        duck.bam.prev_torque = np.zeros(14)
        duck.reset()
    m = duck.model
    duck.soles = [m.geom("left_foot_collision").id, m.geom("right_foot_collision").id]
    duck.head_body = int(m.jnt_bodyid[m.joint("head_roll").id])
    duck.base_mass = float(m.body_mass[duck.trunk])
    duck.base_inertia = m.body_inertia[duck.trunk].copy()
    return duck


# ---- per-episode draws -------------------------------------------------------------------------------------------
def draw_episode(rng: np.random.Generator, episode_s: float, push: dict) -> dict:
    """Dare order: dx, dy, dyaw, sole friction, trunk mass; then the push schedule."""
    d = dict(dx=rng.uniform(-DR["spawn_xy"], DR["spawn_xy"]), dy=rng.uniform(-DR["spawn_xy"], DR["spawn_xy"]),
             dyaw=rng.uniform(-DR["spawn_yaw"], DR["spawn_yaw"]), friction=rng.uniform(*DR["sole_friction"]),
             mass=rng.uniform(*DR["trunk_mass"]))
    pushes, t = {}, rng.uniform(*push["first_s"])
    end = int(round(episode_s / DT))
    while t < episode_s:
        direction, speed = rng.uniform(-math.pi, math.pi), rng.uniform(*push["speed"])
        k = int(round(t / DT))
        if 1 <= k < end:
            pushes[k] = (speed * math.cos(direction), speed * math.sin(direction))
        t += rng.uniform(*push["every_s"])
    d["pushes"] = pushes
    return d


def apply_episode(duck: DuckRefBam, ep: dict, joint_noise: float = 0.0, rng=None) -> None:
    """Reset to the STAND/default pose and apply the draws (episode.ts order: model DR, mj_setConst, spawn pose)."""
    m = duck.model
    for g in duck.soles:
        m.geom_friction[g, 0] = ep["friction"]
    m.body_mass[duck.trunk] = duck.base_mass * ep["mass"]
    m.body_inertia[duck.trunk] = duck.base_inertia * ep["mass"]
    mujoco.mj_setConst(m, duck.data)
    duck.reset()
    q = duck.data.qpos
    q[0] += ep["dx"]
    q[1] += ep["dy"]
    q[3:7] = [math.cos(ep["dyaw"] / 2), 0.0, 0.0, math.sin(ep["dyaw"] / 2)]
    if joint_noise > 0:
        q[duck.qadr] += rng.uniform(-joint_noise, joint_noise, 14)


def apply_push(duck: DuckRefBam, ep: dict, k: int) -> bool:
    p = ep["pushes"].get(k)
    if p is None:
        return False
    duck.data.qvel[0] += p[0]
    duck.data.qvel[1] += p[1]
    return True


# ---- duck state (the only thing reward code sees) -------------------------------------------------------------------
FIELDS = [("lin_vel", 3), ("ang_vel", 3), ("gyro", 3), ("gravity", 3), ("height", 1), ("tilt", 1), ("pitch", 1),
          ("roll", 1), ("joint_pos", 14), ("joint_vel", 14), ("head", 4), ("head_z", 1), ("feet_contact", 2),
          ("feet_z", 2), ("action", 14), ("prev_action", 14), ("command", 13), ("t", 1), ("phase", 1), ("disp", 2),
          ("prev_disp", 2), ("yaw", 1), ("yaw_cum", 1), ("prev_yaw_cum", 1), ("prev_height", 1), ("prev_tilt", 1),
          ("power", 1), ("fell", 1), ("terminated", 1), ("pushed", 1)]
LAYOUT, _o = {}, 0
for _n, _w in FIELDS:
    LAYOUT[_n] = (_o, _w)
    _o += _w
STATE_DIM = _o


def yaw_of(q) -> float:
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Tracker:
    """Per-episode bookkeeping for the state fields that need history (spawn frame, cumulative yaw, prev values)."""

    def __init__(self, duck, ep_s: float):
        d = duck.data
        mujoco.mj_forward(duck.model, d)
        self.x0, self.y0 = float(d.qpos[0]), float(d.qpos[1])
        self.yaw0 = self.yaw_prev = yaw_of(d.qpos[3:7])
        self.yaw_cum, self.ep_s = 0.0, ep_s
        self.prev = dict(disp=np.zeros(2), yaw_cum=0.0, height=float(d.qpos[2]), tilt=0.0)

    def state(self, duck, k, action, prev_action, cmd, fell, terminated, pushed) -> np.ndarray:
        """Call right after mj_forward on the post-step state."""
        d, m = duck.data, duck.model
        s = np.empty(STATE_DIM, np.float32)
        R = d.xmat[duck.trunk].reshape(3, 3)
        q = d.qpos
        yaw = yaw_of(q[3:7])
        c, sn = math.cos(yaw), math.sin(yaw)
        H = np.array([[c, sn, 0.0], [-sn, c, 0.0], [0.0, 0.0, 1.0]])  # world -> heading frame
        w_world = R @ d.qvel[3:6]
        dyaw = wrap(yaw - self.yaw_prev)
        self.yaw_cum += dyaw
        self.yaw_prev = yaw
        c0, s0 = math.cos(self.yaw0), math.sin(self.yaw0)
        dx, dy = q[0] - self.x0, q[1] - self.y0
        disp = np.array([c0 * dx + s0 * dy, -s0 * dx + c0 * dy])
        tilt = math.acos(max(-1.0, min(1.0, R[2, 2])))
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        contact = np.zeros(2)
        if d.ncon:
            g = d.contact.geom[:d.ncon]
            for i, sole in enumerate(duck.soles):
                contact[i] = float(np.any(g == sole))
        tau = duck.bam.prev_torque
        vals = dict(lin_vel=H @ d.qvel[0:3], ang_vel=H @ w_world, gyro=d.sensordata[duck.gyro_adr:duck.gyro_adr + 3],
                    gravity=grav, height=q[2], tilt=tilt, pitch=math.asin(max(-1.0, min(1.0, -R[2, 0]))),
                    roll=math.atan2(R[2, 1], R[2, 2]), joint_pos=q[duck.qadr] - duck.default_pos,
                    joint_vel=d.qvel[duck.vadr], head=q[duck.qadr[5:9]], head_z=d.xpos[duck.head_body, 2],
                    feet_contact=contact, feet_z=d.geom_xpos[duck.soles, 2], action=action, prev_action=prev_action,
                    command=cmd, t=k * DT, phase=min(1.0, k * DT / self.ep_s), disp=disp, prev_disp=self.prev["disp"],
                    yaw=wrap(yaw - self.yaw0), yaw_cum=self.yaw_cum, prev_yaw_cum=self.prev["yaw_cum"],
                    prev_height=self.prev["height"], prev_tilt=self.prev["tilt"],
                    power=float(np.abs(tau * d.qvel[duck.vadr]).sum()), fell=float(fell),
                    terminated=float(terminated), pushed=float(pushed))
        for n, (o, w) in LAYOUT.items():
            s[o:o + w] = vals[n]
        self.prev = dict(disp=disp, yaw_cum=self.yaw_cum, height=float(q[2]), tilt=tilt)
        return s


def unpack(S: np.ndarray) -> dict:
    """[N, STATE_DIM] -> the duck-state dict of [N] / [N, w] arrays (+ a few named conveniences)."""
    out = {}
    for n, (o, w) in LAYOUT.items():
        out[n] = S[:, o] if w == 1 else S[:, o:o + w]
    out["vx"], out["vy"], out["vz"] = out["lin_vel"][:, 0], out["lin_vel"][:, 1], out["lin_vel"][:, 2]
    out["yaw_rate"] = out["ang_vel"][:, 2]
    out["upright"] = -out["gravity"][:, 2]
    for i, n in enumerate(["neck_pitch", "head_pitch", "head_yaw", "head_roll"]):
        out[n] = out["head"][:, i]
    return out


def is_fall(duck) -> bool:
    R8 = duck.data.xmat[duck.trunk][8]
    return math.degrees(math.acos(max(-1.0, min(1.0, R8)))) > FALL_TILT_DEG or duck.data.qpos[2] < FALL_Z


def extra_termination(st: np.ndarray, rules: dict) -> bool:
    """The termination catalogue a move may add (all stricter than the fixed fall)."""
    if not rules:
        return False
    L = LAYOUT
    if "max_tilt_deg" in rules and math.degrees(st[L["tilt"][0]]) > rules["max_tilt_deg"]:
        return True
    if "min_height" in rules and st[L["height"][0]] < rules["min_height"]:
        return True
    o = L["disp"][0]
    if "max_disp_m" in rules and math.hypot(st[o], st[o + 1]) > rules["max_disp_m"]:
        return True
    if "max_heading_drift_deg" in rules and abs(math.degrees(st[L["yaw"][0]])) > rules["max_heading_drift_deg"]:
        return True
    return False


# ---- the training env (one duck) ------------------------------------------------------------------------------------
class DuckEnv:
    """reset() -> (obs61, priv); step(a) -> (obs61, priv, state, terminated, timeout). Auto-reset is the caller's job."""

    PRIV = 6

    def __init__(self, policy: str, cmd_lo, cmd_hi, episode_s: float, terminations: dict | None, seed: int,
                 push: dict = PUSH_TRAIN, joint_noise: float = RESET_JOINT_NOISE, bam: str = "numba"):
        self.duck = make_duck(bam)
        self.policy, self.lo, self.hi = policy, np.asarray(cmd_lo, float), np.asarray(cmd_hi, float)
        self.episode_s, self.ep_len = episode_s, int(round(episode_s / DT))
        self.rules, self.push, self.noise = terminations or {}, push, joint_noise
        self.rng = np.random.default_rng(seed)

    def reset(self, ep: dict | None = None, cmd=None):
        self.ep = ep if ep is not None else draw_episode(self.rng, self.episode_s, self.push)
        apply_episode(self.duck, self.ep, self.noise if ep is None else 0.0, self.rng)
        self.cmd = np.asarray(cmd, float) if cmd is not None else self.rng.uniform(self.lo, self.hi)
        self.k = 0
        self.tracker = Tracker(self.duck, self.episode_s)
        return self._obs()

    def _obs(self):
        duck = self.duck
        mujoco.mj_forward(duck.model, duck.data)
        c = self.cmd
        obs, parts = duck.build_obs(self.policy, c[0:3], c[3:7], c[7:13])
        self._vel_now = parts["vel_now"]
        d = duck.data
        yaw = yaw_of(d.qpos[3:7])
        cy, sy = math.cos(yaw), math.sin(yaw)
        v = d.qvel
        priv = np.array([2 * (cy * v[0] + sy * v[1]), 2 * (-sy * v[0] + cy * v[1]), 2 * v[2],
                         10 * (d.qpos[2] - STAND_Z), self.ep["friction"] - 1.0, 10 * (self.ep["mass"] - 1.0)], np.float32)
        return obs.astype(np.float32), priv

    def step(self, action):
        duck = self.duck
        action = np.asarray(action, np.float32)
        prev = duck.prev_action.copy()
        duck.substeps(duck.default_pos + 1.0 * action.astype(np.float64))
        duck.prev_action = action
        duck.lag_joint_vel = self._vel_now
        self.k += 1
        pushed = apply_push(duck, self.ep, self.k)
        obs, priv = self._obs()  # includes mj_forward
        fell = is_fall(duck)
        st = self.tracker.state(duck, self.k, action, prev, self.cmd, fell, fell, pushed)
        term = fell or extra_termination(st, self.rules)
        st[LAYOUT["terminated"][0]] = float(term)
        timeout = (not term) and self.k >= self.ep_len
        return obs, priv, st, term, timeout
