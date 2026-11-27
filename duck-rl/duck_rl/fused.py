"""The fused per-tick kernel of the fast rollout (kit 1.8.0; spikes/physics-speedup/fast/README.md lever 3): one nogil
numba call per 50-Hz tick runs the tick's 4 substeps with no Python in between, bit-identical to the stock loop.

  solo Challenge (env.ChallengeEnv.step):  4 x [BAM (lab.duck._bam_substep) -> effort (env._effort) -> mj_step -> floor-out check]
  duel Challenge (duel.DuelWorld.step): 4 x [BAM slot 0 -> BAM slot 1 -> mj_step]

The BAM and the effort are the stock njit functions themselves (called from the kernel, so the arithmetic is the same
code); `mj_step` is MuJoCo's C function called through ctypes from the library the Python bindings load; the BAM's
efc_* rows and the contact list are read from mjData through struct offsets (OFFSETS: offsetof from mujoco's
include/mujoco/mjdata.h, pinned per mujoco version). Before an env's first fused tick, verify() checks the offsets
against the bindings' own views on a state with constraint rows and contacts (until then the env runs the stock loop;
on a mismatch it stays on the stock loop for good, with a warning). Everything else in a step is the stock Python.

available() says whether the kernel can run here (numba, a pinned mujoco version, the library and its mj_step symbol);
when it cannot, the envs run the stock loop (the trainer prints why). The kernel compiles once per worker process
(~1-2 s at the start; not cached: it embeds the library's function pointer).
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys

import mujoco
import numpy as np

# mjData / mjContact layout per mujoco version (offsetof in include/mujoco/mjdata.h; LP64). verify() checks them.
OFFSETS = {
    "3.14.0": dict(nefc=160576, efc_type=161456, efc_id=161464, efc_force=162056, ncon=160560, contact=161448,
                   sizeof_contact=584, contact_geom=540),
}
_O_KEYS = ("nefc", "efc_type", "efc_id", "efc_force", "ncon", "contact", "sizeof_contact", "contact_geom")

_STATE: dict = {}  # "ok": bool, "reason": str, "O": int64 offsets, "tick"/"tick2": the kernels
_WARNED: set = set()


def _warn(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f"[duck-rl] fast rollout: {msg}", file=sys.stderr, flush=True)


def _libmujoco():
    """The MuJoCo shared library the bindings load (the wheel ships it beside the package)."""
    d = os.path.dirname(mujoco.__file__)
    for pat in ("libmujoco*.dylib", "libmujoco.so*", "mujoco*.dll"):
        for p in sorted(glob.glob(os.path.join(d, pat))):
            return ctypes.CDLL(p)
    raise OSError(f"no libmujoco in {d}")


def available() -> tuple[bool, str]:
    """(True, "") when the fused kernel can run in this process, else (False, why). Builds the kernels on first call."""
    if "ok" in _STATE:
        return _STATE["ok"], _STATE["reason"]
    try:
        off = OFFSETS.get(mujoco.__version__)
        if off is None:
            raise RuntimeError(f"no pinned mjData offsets for mujoco {mujoco.__version__} (pinned: {', '.join(OFFSETS)})")
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            raise RuntimeError("not a 64-bit process")
        lib = _libmujoco()
        c_step = lib.mj_step  # AttributeError when the symbol is missing
        c_step.argtypes, c_step.restype = [ctypes.c_void_p, ctypes.c_void_p], None
        _STATE.update(_build(c_step))
        _STATE["O"] = np.array([off[k] for k in _O_KEYS], np.int64)
        _STATE["ok"], _STATE["reason"] = True, ""
    except Exception as e:  # noqa: BLE001  (no numba, no library, no symbol, an unpinned mujoco: the stock loop runs)
        _STATE["ok"], _STATE["reason"] = False, f"{type(e).__name__}: {e}"
    return _STATE["ok"], _STATE["reason"]


def _build(c_step) -> dict:
    """The numba kernels (imported here: a machine without numba falls back in available())."""
    import numba
    from numba import njit, types
    from numba.extending import intrinsic

    from lab.duck import _bam_substep

    from .env import _effort

    def mk_ptr(t):
        @intrinsic
        def to_ptr(typingctx, addr):
            sig = types.CPointer(t)(types.int64)

            def codegen(context, builder, signature, args):
                return builder.inttoptr(args[0], context.get_value_type(signature.return_type))
            return sig, codegen
        return to_ptr

    pf64, pi32, pi64 = mk_ptr(types.float64), mk_ptr(types.int32), mk_ptr(types.int64)

    @njit(nogil=True)
    def i32_at(addr):
        return numba.carray(pi32(addr), 1)[0]

    @njit(nogil=True)
    def i64_at(addr):
        return numba.carray(pi64(addr), 1)[0]

    @njit(nogil=True)
    def bam(d_addr, O, ptrs, nv, ring, head, delay, targets, qadr, dof, prev_torque, P, fric_type, act):
        """One BAM substep (lab.duck NumbaBam.before_substep) on the mjData at d_addr."""
        nefc = i32_at(d_addr + O[0])
        efc_type = numba.carray(pi32(i64_at(d_addr + O[1])), nefc)
        efc_id = numba.carray(pi32(i64_at(d_addr + O[2])), nefc)
        efc_force = numba.carray(pf64(i64_at(d_addr + O[3])), nefc)
        qpos = numba.carray(pf64(ptrs[0]), ptrs[8])
        qvel = numba.carray(pf64(ptrs[1]), nv)
        return _bam_substep(ring, head, delay, targets, qpos, qvel, qadr, dof, prev_torque, P, efc_type, efc_id,
                            efc_force, nefc, fric_type, numba.carray(pf64(ptrs[2]), nv), numba.carray(pf64(ptrs[3]), nv),
                            numba.carray(pf64(ptrs[4]), nv), numba.carray(pf64(ptrs[6]), nv),
                            numba.carray(pf64(ptrs[5]), ptrs[7]), act, nv)

    # ptrs: 0 qpos, 1 qvel, 2 qfrc_bias, 3 qfrc_constraint, 4 qfrc_actuator, 5 ctrl, 6 dof_frictionloss (the model's),
    #       7 nu, 8 nq (counts, not pointers)
    # I:    qadr[0:14], dof[14:28], act[28:42], legs[42:52], vadr_legs[52:62], duck_mask[62:62 + ngeom]
    # K:    0 m_addr, 1 d_addr, 2 nv, 3 ngeom, 4 delay, 5 fric_type, 6 floor, 7 nsub
    @njit(nogil=True)
    def tick(ptrs, I, K, O, ring, head, targets, prev_torque, P, acc, tlim, check):
        """ChallengeEnv.step's substep loop: nsub x (BAM, _effort, mj_step, the floor-out check). The check reads mjData's
        contact list after mj_step, as the stock loop does (the contacts mj_step found for the state it started from).
        -> (head, the first substep whose check touched the floor, else -1)."""
        m_addr, d_addr, nv, ngeom = K[0], K[1], K[2], K[3]
        delay, fric_type, floor, nsub = K[4], K[5], K[6], K[7]
        qadr, dof, act = I[0:14], I[14:28], I[28:42]
        legs, vadr_legs, mask = I[42:52], I[52:62], I[62:62 + ngeom]
        qvel = numba.carray(pf64(ptrs[1]), nv)
        touched = -1
        for s in range(nsub):
            head = bam(d_addr, O, ptrs, nv, ring, head, delay, targets, qadr, dof, prev_torque, P, fric_type, act)
            _effort(acc, prev_torque, qvel, legs, vadr_legs, tlim)
            c_step(m_addr, d_addr)
            if check and touched < 0:
                ncon = i32_at(d_addr + O[4])
                if ncon > 0:
                    base = i64_at(d_addr + O[5])
                    for c in range(ncon):
                        g = numba.carray(pi32(base + c * O[6] + O[7]), 2)
                        g1, g2 = g[0], g[1]
                        if (g1 == floor and mask[g2] != 0) or (g2 == floor and mask[g1] != 0):
                            touched = s
                            break
        return head, touched

    # duel: I0 / I1 = each slot's qadr[0:14], dof[14:28], act[28:42]; K: 0 m_addr, 1 d_addr, 2 nv, 3 delay,
    # 4 fric_type, 5 nsub
    @njit(nogil=True)
    def tick2(ptrs, I0, I1, K, O, ring0, head0, targets0, prev0, P0, ring1, head1, targets1, prev1, P1):
        """DuelWorld.step's substep loop: nsub x (BAM slot 0, BAM slot 1, mj_step). -> (head0, head1)."""
        m_addr, d_addr, nv, delay, fric_type, nsub = K[0], K[1], K[2], K[3], K[4], K[5]
        for s in range(nsub):
            head0 = bam(d_addr, O, ptrs, nv, ring0, head0, delay, targets0, I0[0:14], I0[14:28], prev0, P0, fric_type,
                        I0[28:42])
            head1 = bam(d_addr, O, ptrs, nv, ring1, head1, delay, targets1, I1[0:14], I1[14:28], prev1, P1, fric_type,
                        I1[28:42])
            c_step(m_addr, d_addr)
        return head0, head1

    return dict(tick=tick, tick2=tick2)


def verify(m: mujoco.MjModel, d: mujoco.MjData) -> bool | None:
    """The pinned offsets agree with the bindings' views: True / False, or None while the state has no constraint row
    or no contact to check them on (the caller runs the stock loop until then)."""
    if not (d.nefc and d.ncon):
        return None
    O = dict(zip(_O_KEYS, _STATE["O"].tolist()))
    base = d._address

    def i32(off):
        return ctypes.c_int32.from_address(base + off).value

    def ptr(off):
        return ctypes.c_int64.from_address(base + off).value
    g = d.contact.geom
    return (i32(O["nefc"]) == d.nefc and i32(O["ncon"]) == d.ncon
            and ptr(O["efc_type"]) == d.efc_type.ctypes.data and ptr(O["efc_id"]) == d.efc_id.ctypes.data
            and ptr(O["efc_force"]) == d.efc_force.ctypes.data
            and ptr(O["contact"]) + O["contact_geom"] == g.ctypes.data and g.strides[0] == O["sizeof_contact"])


def _ptrs(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    return np.array([d.qpos.ctypes.data, d.qvel.ctypes.data, d.qfrc_bias.ctypes.data, d.qfrc_constraint.ctypes.data,
                     d.qfrc_actuator.ctypes.data, d.ctrl.ctypes.data, m.dof_frictionloss.ctypes.data, m.nu, m.nq], np.int64)


class _Gate:
    """The per-env state of the layout check: None = not checked yet (stock loop), True = fused, False = stock for good."""

    def __init__(self):
        self.ok = None

    def ready(self, m, d) -> bool:
        if self.ok and not _STATE["ok"]:  # the kernel failed to compile in this process (_call)
            self.ok = False
        if self.ok is None:
            v = verify(m, d)
            if v is False:
                _warn(f"mjData offsets do not match mujoco {mujoco.__version__}'s bindings: this env runs the stock loop")
            self.ok = v
        return bool(self.ok)

    def _call(self, *args):
        """self.fn(*args), or None when the kernel fails to compile on its first call in this process (numba typing or
        lowering errors are raised before the kernel runs, so nothing has been stepped): the kernel is then off for
        every env of this process, which run the stock loop (the pool reports the fused env count to the trainer)."""
        if _STATE.get("compiled"):
            return self.fn(*args)
        try:
            out = self.fn(*args)
        except Exception as e:  # noqa: BLE001
            _STATE["ok"], _STATE["reason"] = False, f"the kernel failed to compile: {type(e).__name__}: {e}"
            self.ok = False
            _warn(f"{_STATE['reason']}: this worker's envs run the stock loop")
            return None
        _STATE["compiled"] = True  # (both kernels compile on their first calls; one process runs one kind of env)
        return out


class SoloTick(_Gate):
    """The fused substep loop of one ChallengeEnv (env.ChallengeEnv.step calls run() when ready())."""

    def __init__(self, env):
        super().__init__()
        from .env import LEGS
        duck = env.duck
        m, d, b = duck.model, duck.data, duck.bam
        assert len(b.qadr_i) == len(b.dof_i) == len(b.act_i) == 14 and len(LEGS) == 10
        self.ptrs = _ptrs(m, d)
        self.I = np.concatenate([b.qadr_i, b.dof_i, b.act_i, LEGS, env._vadr_legs,
                                 env.duck_mask.astype(np.int64)]).astype(np.int64)
        from duckref.bam import FRICTION_DOF
        self.K = np.array([m._address, d._address, m.nv, m.ngeom, b.cfg.delay_substeps, int(FRICTION_DOF), env.floor,
                           duck.decimation], np.int64)
        self.O, self.fn = _STATE["O"], _STATE["tick"]

    def run(self, bam, acc, tlim, check) -> int | None:
        """The tick's substeps; bam.head updated in place. -> the first substep that touched the floor, else -1; None
        when the kernel could not compile (nothing stepped: the caller runs the stock loop)."""
        r = self._call(self.ptrs, self.I, self.K, self.O, bam.ring, bam.head, bam.targets, bam.prev_torque, bam.P, acc,
                       tlim, check)
        if r is None:
            return None
        bam.head, s = r
        return s


class DuelTick(_Gate):
    """The fused substep loop of one DuelWorld (duel.DuelWorld.step calls run() when ready())."""

    def __init__(self, world):
        super().__init__()
        from duckref.bam import FRICTION_DOF
        m, d = world.m, world.d
        b0, b1 = world.bams
        assert b0.cfg.delay_substeps == b1.cfg.delay_substeps
        self.ptrs = _ptrs(m, d)
        self.I = [np.concatenate([b.qadr_i, b.dof_i, b.act_i]).astype(np.int64) for b in (b0, b1)]
        assert all(len(i) == 42 for i in self.I)
        self.K = np.array([m._address, d._address, m.nv, b0.cfg.delay_substeps, int(FRICTION_DOF), world.decimation],
                          np.int64)
        self.O, self.fn = _STATE["O"], _STATE["tick2"]

    def run(self, b0, b1) -> bool:
        """The tick's substeps; the heads updated in place. -> False when the kernel could not compile (nothing
        stepped: the caller runs the stock loop)."""
        r = self._call(self.ptrs, self.I[0], self.I[1], self.K, self.O, b0.ring, b0.head, b0.targets, b0.prev_torque,
                       b0.P, b1.ring, b1.head, b1.targets, b1.prev_torque, b1.P)
        if r is None:
            return False
        b0.head, b1.head = r
        return True


def solo_tick(env):
    """A SoloTick for env when the kernel is available, else None (the stock loop)."""
    ok, why = available()
    if not ok:
        _warn(f"the fused kernel is off ({why}): the stock substep loop runs")
        return None
    return SoloTick(env)


def duel_tick(world):
    ok, why = available()
    if not ok:
        _warn(f"the fused kernel is off ({why}): the stock substep loop runs")
        return None
    return DuelTick(world)
