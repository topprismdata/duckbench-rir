"""Plant v1 reference: the BAM M6 XL330 actuator on top of the plant v0 loop (docs/spikes/D24-plant-v1.md §2).

Mirrors bam/mjlab.py BamActuator (Rhoban/bam @62bd8ce, the GPU actuator microduck_rl trained with) on CPU MuJoCo, per
physics substep, before each mj_step and reading the data the previous mj_step (or the tick's mj_forward) left:
  delay     the position target is delayed by a fixed number of substeps (mjlab DelayBuffer: append, then read lag);
  voltage   vin = max(vin_min, vin0 - gain * sum_j |tau_prev_j|), tau_prev = last substep's unclamped motor torque;
  control   duty = (qt - q) * kp_fw * error_gain, clamped to the current-limit window, then to ±max_pwm
            (bam/actuator.py:251-269); torque = kt * (vin * duty) / R - kt^2 * dq / R (bam/actuator.py:289-292) -> ctrl;
  friction  the m6 budget (bam/mjlab.py:425-496) from tau_m = qfrc_actuator (previous solve) and
            tau_ext = -qfrc_bias + qfrc_constraint - own DOF-friction rows -> dof_frictionloss (x friction_scale).
Every sum runs in joint order, so the TS port (packages/sim/src/actuator/bam.ts) can reproduce it bit for bit, except
where libm differs (exp, pow in the Stribeck term).
"""
from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from .plant import REPO, DuckRef

# DUCKBENCH_PLANT_DIR overrides the bundle (a copy of duckref outside the repo: the RL kit for players)
V1_BUNDLE_DIR = Path(os.environ.get("DUCKBENCH_PLANT_DIR") or REPO / "packages" / "sim" / "plant" / "microduck-legs-bam-v1")
FRICTION_DOF = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)

BAM_PARAM_KEYS = ["kt", "R", "armature", "friction_base", "friction_stribeck", "load_friction_motor",
                  "load_friction_external", "load_friction_motor_stribeck", "load_friction_external_stribeck",
                  "load_friction_motor_quad", "load_friction_external_quad", "dtheta_stribeck", "alpha",
                  "friction_viscous"]


def bam_library(src: str | Path):
    """The real bam library's XL330 M6 model (CPU classes, no torch; colorama stubbed), for cross-checks."""
    c = types.ModuleType("colorama")

    class _Blank:
        def __getattr__(self, _k):
            return ""

    c.Fore = c.Back = c.Style = _Blank()
    c.init = lambda *a, **k: None
    sys.modules.setdefault("colorama", c)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from bam.model import load_model

    model = load_model(motor_name="xl330", model="m6")
    assert model.stribeck and model.load_dependent and model.directional and model.quadratic, "not the m6 form"
    return model


@dataclass(frozen=True)
class BamConfig:
    p: dict
    kp_fw: float
    error_gain: float
    max_pwm: float
    max_current: float
    vin: float
    vin_drop_gain: float
    vin_min: float
    delay_substeps: int
    friction_scale: float

    @classmethod
    def from_block(cls, blk: dict, **overrides) -> "BamConfig":
        c = cls(p=dict(blk["params"]), kp_fw=blk["kp_fw"], error_gain=blk["error_gain"], max_pwm=blk["max_pwm"],
                max_current=blk["max_current"], vin=blk["vin"], vin_drop_gain=blk["vin_drop_gain"],
                vin_min=blk["vin_min"], delay_substeps=int(blk["delay_substeps"]),
                friction_scale=blk["friction_scale"])
        return replace(c, **overrides)


def supply_voltage(c: BamConfig, prev_torque: np.ndarray) -> float:
    """bam/mjlab.py:601-607. The load sum runs in joint order (not numpy's pairwise sum)."""
    load = 0.0
    for t in prev_torque:
        load += abs(float(t))
    return max(c.vin - c.vin_drop_gain * load, c.vin_min)


def motor_torque(c: BamConfig, vin: float, qt, q, dq):
    """Firmware P law + current limiter + PWM clamp + DC motor (bam/actuator.py:251-292), same operation order."""
    kt, R = c.p["kt"], c.p["R"]
    duty = (qt - q) * c.kp_fw * c.error_gain
    back_emf = kt * dq
    span = R * c.max_current / vin
    center = back_emf / vin
    duty = np.minimum(np.maximum(duty, center - span), center + span)
    duty = np.minimum(np.maximum(duty, -c.max_pwm), c.max_pwm)
    volts = vin * duty
    torque = kt * volts / R
    return torque - (kt * kt) * dq / R


def stribeck(c: BamConfig, dq):
    return np.exp(-np.power(np.abs(dq) / c.p["dtheta_stribeck"], c.p["alpha"]))


def friction_budget(c: BamConfig, tau_m, tau_ext, dq):
    """m6 velocity-independent friction budget, GPU form (bam/mjlab.py:442-496) x friction_scale."""
    p = c.p
    s = stribeck(c, dq)
    fl = p["friction_base"] + s * p["friction_stribeck"]
    fl = fl + np.abs(tau_ext * p["load_friction_external"] - tau_m * p["load_friction_motor"])
    fl = fl + s * np.abs(tau_ext * p["load_friction_external_stribeck"] - tau_m * p["load_friction_motor_stribeck"])
    a_ext, a_mot = np.abs(tau_ext), np.abs(tau_m)
    quad = np.where(a_mot > a_ext, p["load_friction_external_quad"] * (a_ext * a_ext),
                    p["load_friction_motor_quad"] * (a_mot * a_mot))
    fl = fl + s * quad
    return fl * c.friction_scale


def external_torque(m: mujoco.MjModel, d: mujoco.MjData, dof: np.ndarray) -> np.ndarray:
    """-qfrc_bias + qfrc_constraint - the force of our own DOF-friction rows (matched by DOF id; bam/mjlab.py:541-645)."""
    fric = np.zeros(m.nv)
    for r in range(d.nefc):
        if d.efc_type[r] == FRICTION_DOF:
            fric[d.efc_id[r]] += d.efc_force[r]
    return -d.qfrc_bias[dof] + d.qfrc_constraint[dof] - fric[dof]


class BamActuator:
    """Per-MjData actuator state: the delay ring, the previous motor torque, the latched position targets."""

    def __init__(self, m: mujoco.MjModel, qadr: np.ndarray, dof: np.ndarray, act: np.ndarray, cfg: BamConfig):
        self.m, self.qadr, self.dof, self.act, self.cfg = m, qadr, dof, act, cfg
        self.size = cfg.delay_substeps + 1
        self.ring = np.zeros((self.size, len(dof)))
        self.head = 0
        self.prev_torque = np.zeros(len(dof))
        self.targets = np.zeros(len(dof))

    def reset(self, d: mujoco.MjData, default_pos: np.ndarray) -> None:
        """Ring back-filled with the reset targets (DEFAULT_POS); ctrl = the torque at q = target, dq = 0 (zero)."""
        self.ring[:] = default_pos
        self.head = 0
        self.prev_torque[:] = 0.0
        self.targets[:] = default_pos
        d.ctrl[self.act] = 0.0
        z = np.zeros(len(self.dof))
        self.m.dof_frictionloss[self.dof] = friction_budget(self.cfg, z, z, z)

    def state(self) -> dict:
        return dict(ring=self.ring.copy(), head=self.head, prev_torque=self.prev_torque.copy())

    def latch(self, targets: np.ndarray) -> None:
        self.targets[:] = targets

    def before_substep(self, d: mujoco.MjData) -> dict:
        c, m, dof = self.cfg, self.m, self.dof
        self.head = (self.head + 1) % self.size
        self.ring[self.head] = self.targets
        qt = self.ring[(self.head - c.delay_substeps) % self.size].copy()
        q, dq = d.qpos[self.qadr].copy(), d.qvel[dof].copy()
        vin = supply_voltage(c, self.prev_torque)
        tau = motor_torque(c, vin, qt, q, dq)
        tau_ext = external_torque(m, d, dof)
        tau_m = d.qfrc_actuator[dof].copy()
        fl = friction_budget(c, tau_m, tau_ext, dq)
        m.dof_frictionloss[dof] = fl
        d.ctrl[self.act] = tau
        self.prev_torque = tau
        return dict(qt=qt, q=q, dq=dq, vin=vin, tau_ext=tau_ext, tau_m=tau_m, torque=tau, frictionloss=fl)


class DuckRefBam(DuckRef):
    """DuckRef on plant v1: same obs/policy loop (plant-v0 §5), BAM between the policy targets and the physics."""

    def __init__(self, bundle_dir: str | Path = V1_BUNDLE_DIR, model: mujoco.MjModel | None = None, **overrides):
        self._overrides = overrides
        super().__init__(bundle_dir, model)

    def reset(self) -> None:
        if not hasattr(self, "bam"):
            self.bam_cfg = BamConfig.from_block(self.manifest["actuator"], **self._overrides)
            self.bam = BamActuator(self.model, self.qadr, self.vadr, self.act, self.bam_cfg)
        super().reset()
        self.bam.reset(self.data, self.default_pos)

    def substeps(self, targets: np.ndarray, record: list | None = None) -> None:
        """Step 5 on plant v1: latch the targets, then BAM + mj_step per substep."""
        self.bam.latch(targets)
        for _ in range(self.decimation):
            rec = self.bam.before_substep(self.data)
            if record is not None:
                record.append(rec)
            mujoco.mj_step(self.model, self.data)

    def tick(self, policy: str, twist3=(0.0, 0.0, 0.0), head_target4=(0.0, 0.0, 0.0, 0.0), body6=None,
             record_substeps: bool = False) -> dict:
        m, d = self.model, self.data
        start = dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), qacc_warmstart=d.qacc_warmstart.copy(),
                     ctrl=d.ctrl[self.act].copy(), frictionloss=m.dof_frictionloss[self.vadr].copy(),
                     **self.bam.state())
        mujoco.mj_forward(m, d)
        obs64, parts = self.build_obs(policy, twist3, head_target4, body6)
        obs = obs64.astype(np.float32)
        action = self.session(policy).run(None, {"obs": obs[None, :]})[0][0].astype(np.float32)
        targets = self.default_pos + 1.0 * action.astype(np.float64)
        subs: list = []
        self.substeps(targets, subs if record_substeps else None)
        self.prev_action = action
        self.lag_joint_vel = parts["vel_now"]
        rec = dict(k=self.k, policy=policy, **start, lag_joint_vel=parts["joint_vel"],
                   prev_action=parts["prev_action"], command13=parts["command13"], obs=obs, action=action,
                   targets=targets, substeps=subs)
        self.k += 1
        return rec

    def hold(self) -> None:
        """One policy tick with the targets frozen (a 'settle' segment): BAM keeps running."""
        self.substeps(self.bam.targets.copy())
        self.k += 1
