"""T2 (Mac-speed research, opt-in): the training env with the obs builder and Tracker.state as numba kernels.

Same plant, BAM, DR, pushes, resets and terminations as lab/duck.py DuckEnv; only the two per-env Python parts that
read MjData after mj_forward are replaced (duckref build_obs + priv: 16 -> 1.6 us; Tracker.state: 36 -> 3 us per env
step, M in docs/research/novel-moves/rl-code/mac-speed/cpu/). Outputs are bit-identical to DuckEnv/Pool (float32
obs, priv, state; the same float64 arithmetic in the same order), checked by mac-speed/t1t2/run.py parity.
FastPool is lab/pool.py Pool with this env in the workers. Nothing in duck.py / pool.py changes.
"""
from __future__ import annotations

import math
import multiprocessing as mp

import mujoco
import numpy as np
from numba import njit

from .duck import DT, LAYOUT, STATE_DIM, DuckEnv, Tracker, apply_episode, apply_push, draw_episode, extra_termination, is_fall
from .pool import Pool

_TERM = LAYOUT["terminated"][0]


@njit(cache=True)
def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@njit(cache=True)
def _yaw(q0, q1, q2, q3):
    return math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))


@njit(cache=True)
def _clip1(x):
    return max(-1.0, min(1.0, x))


@njit(cache=True)
def nb_obs(obs, priv, vel_now, head_cmd, sensordata, gyro_adr, xquat, qpos, qvel, qadr, vadr, default_pos, lag,
           has_lag, prev_action, cmd, use_head, use_body, fric, mass):
    """duckref DuckRef.build_obs (float64, cast to float32 once) + DuckEnv._obs priv. Mutates head_cmd (EMA)."""
    for i in range(3):
        obs[i] = sensordata[gyro_adr + i]
    q0, q1, q2, q3 = xquat[0], xquat[1], xquat[2], xquat[3]
    if q0 == 1 and q1 == 0 and q2 == 0 and q3 == 0:  # mju_quat2Mat's identity shortcut
        r6, r7, r8 = 0.0, 0.0, 1.0
    else:
        r6 = 2 * (q1 * q3 - q0 * q2)
        r7 = 2 * (q2 * q3 + q0 * q1)
        r8 = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3
    obs[3], obs[4], obs[5] = -r6, -r7, -r8  # R^T (0, 0, -1)
    for j in range(14):
        obs[6 + j] = qpos[qadr[j]] - default_pos[j]
        vel_now[j] = qvel[vadr[j]]
        obs[20 + j] = lag[j] if has_lag else vel_now[j]
        obs[34 + j] = prev_action[j]
    for i in range(3):
        obs[48 + i] = cmd[i]
    for i in range(4):
        head_cmd[i] = head_cmd[i] + 0.2 * (cmd[3 + i] - head_cmd[i])
        obs[51 + i] = head_cmd[i] if use_head else 0.0
    for i in range(6):
        obs[55 + i] = cmd[7 + i] if use_body else 0.0
    yaw = _yaw(qpos[3], qpos[4], qpos[5], qpos[6])
    cy, sy = math.cos(yaw), math.sin(yaw)
    priv[0] = 2 * (cy * qvel[0] + sy * qvel[1])
    priv[1] = 2 * (-sy * qvel[0] + cy * qvel[1])
    priv[2] = 2 * qvel[2]
    priv[3] = 10 * (qpos[2] - 0.12)  # STAND_Z
    priv[4] = fric - 1.0
    priv[5] = 10 * (mass - 1.0)


@njit(cache=True)
def nb_state(s, trk, xmat, qpos, qvel, sensordata, gyro_adr, head_z, geom_xpos, soles, cgeom, ncon, tau, qadr, vadr,
             default_pos, action, prev_action, cmd, k, dt, ep_s, fell, terminated, pushed):
    """lab.duck.Tracker.state in the FIELDS order. trk = [x0, y0, yaw0, yaw_prev, yaw_cum, prev_disp0, prev_disp1,
    prev_yaw_cum, prev_height, prev_tilt] (the Tracker's history, updated in place)."""
    R = xmat
    yaw = _yaw(qpos[3], qpos[4], qpos[5], qpos[6])
    c, sn = math.cos(yaw), math.sin(yaw)
    w0 = R[0] * qvel[3] + R[1] * qvel[4] + R[2] * qvel[5]
    w1 = R[3] * qvel[3] + R[4] * qvel[4] + R[5] * qvel[5]
    w2 = R[6] * qvel[3] + R[7] * qvel[4] + R[8] * qvel[5]
    dyaw = _wrap(yaw - trk[3])
    trk[4] += dyaw
    trk[3] = yaw
    c0, s0 = math.cos(trk[2]), math.sin(trk[2])
    dx, dy = qpos[0] - trk[0], qpos[1] - trk[1]
    d0, d1 = c0 * dx + s0 * dy, -s0 * dx + c0 * dy
    tilt = math.acos(_clip1(R[8]))
    i = 0
    # lin_vel = H qvel[0:3], ang_vel = H w, H = [[c, sn, 0], [-sn, c, 0], [0, 0, 1]] (the 0.0 * z terms keep numpy's
    # matmul arithmetic, so the float32 result is identical)
    for a, b, z in ((qvel[0], qvel[1], qvel[2]), (w0, w1, w2)):
        s[i] = c * a + sn * b + 0.0 * z
        s[i + 1] = -sn * a + c * b + 0.0 * z
        s[i + 2] = 0.0 * a + 0.0 * b + 1.0 * z
        i += 3
    for j in range(3):
        s[i + j] = sensordata[gyro_adr + j]
    i += 3
    s[i], s[i + 1], s[i + 2] = -R[6], -R[7], -R[8]  # gravity = R^T (0, 0, -1)
    i += 3
    s[i] = qpos[2]
    s[i + 1] = tilt
    s[i + 2] = math.asin(_clip1(-R[6]))
    s[i + 3] = math.atan2(R[7], R[8])
    i += 4
    for j in range(14):
        s[i + j] = qpos[qadr[j]] - default_pos[j]
        s[i + 14 + j] = qvel[vadr[j]]
    i += 28
    for j in range(4):
        s[i + j] = qpos[qadr[5 + j]]
    i += 4
    s[i] = head_z
    i += 1
    for f in range(2):
        hit = 0.0
        for r in range(ncon):
            if cgeom[r, 0] == soles[f] or cgeom[r, 1] == soles[f]:
                hit = 1.0
                break
        s[i + f] = hit
        s[i + 2 + f] = geom_xpos[soles[f], 2]
    i += 4
    for j in range(14):
        s[i + j] = action[j]
        s[i + 14 + j] = prev_action[j]
    i += 28
    for j in range(13):
        s[i + j] = cmd[j]
    i += 13
    s[i] = k * dt
    s[i + 1] = min(1.0, k * dt / ep_s)
    s[i + 2], s[i + 3] = d0, d1
    s[i + 4], s[i + 5] = trk[5], trk[6]
    s[i + 6] = _wrap(yaw - trk[2])
    s[i + 7] = trk[4]
    s[i + 8] = trk[7]
    s[i + 9] = trk[8]
    s[i + 10] = trk[9]
    # power = np.abs(tau * qvel[vadr]).sum(): numpy's pairwise sum for n = 14 (8 partials, then the tail in order)
    p = np.empty(14)
    for j in range(14):
        p[j] = abs(tau[j] * qvel[vadr[j]])
    pw = ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]))
    for j in range(8, 14):
        pw += p[j]
    s[i + 11] = pw
    s[i + 12], s[i + 13], s[i + 14] = fell, terminated, pushed
    trk[5], trk[6], trk[7], trk[8], trk[9] = d0, d1, trk[4], qpos[2], tilt


class FastDuckEnv(DuckEnv):
    """DuckEnv with nb_obs / nb_state. The Tracker object is still built at reset (spawn frame, one mj_forward, as in
    DuckEnv); its history then lives in self._trk."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        duck = self.duck
        names = duck.command_names(self.policy)
        self._use_head, self._use_body = "head_pose" in names, "body_pose" in names
        self._soles = np.asarray(duck.soles, np.int64)
        self._qadr, self._vadr = np.asarray(duck.qadr, np.int64), np.asarray(duck.vadr, np.int64)
        self._ob, self._pr, self._vn = np.empty(61, np.float32), np.empty(6, np.float32), np.empty(14)
        self._trk = np.zeros(10)

    def reset(self, ep=None, cmd=None):
        self.ep = ep if ep is not None else draw_episode(self.rng, self.episode_s, self.push)
        apply_episode(self.duck, self.ep, self.noise if ep is None else 0.0, self.rng)
        self.cmd = np.asarray(cmd, float) if cmd is not None else self.rng.uniform(self.lo, self.hi)
        self.k = 0
        tr = self.tracker = Tracker(self.duck, self.episode_s)
        self._trk[:] = [tr.x0, tr.y0, tr.yaw0, tr.yaw_prev, tr.yaw_cum, 0.0, 0.0, 0.0, tr.prev["height"], 0.0]
        return self._obs()

    def _obs(self):
        duck = self.duck
        d = duck.data
        mujoco.mj_forward(duck.model, d)
        lag = duck.lag_joint_vel
        nb_obs(self._ob, self._pr, self._vn, duck.head_cmd, d.sensordata, duck.gyro_adr, d.xquat[duck.trunk], d.qpos,
               d.qvel, self._qadr, self._vadr, duck.default_pos, lag if lag is not None else self._vn, lag is not None,
               duck.prev_action, self.cmd, self._use_head, self._use_body, self.ep["friction"], self.ep["mass"])
        self._vel_now = self._vn.copy()  # becomes duck.lag_joint_vel next tick (build_obs's parts["vel_now"])
        return self._ob.copy(), self._pr.copy()

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
        d = duck.data
        st = np.empty(STATE_DIM, np.float32)
        nb_state(st, self._trk, d.xmat[duck.trunk], d.qpos, d.qvel, d.sensordata, duck.gyro_adr,
                 d.xpos[duck.head_body, 2], d.geom_xpos, self._soles, d.contact.geom, d.ncon, duck.bam.prev_torque,
                 self._qadr, self._vadr, duck.default_pos, action, prev, self.cmd, self.k, DT, self.episode_s,
                 float(fell), float(fell), float(pushed))
        term = fell or extra_termination(st, self.rules)
        st[_TERM] = float(term)
        timeout = (not term) and self.k >= self.ep_len
        return obs, priv, st, term, timeout

    @property
    def disp_now(self) -> float:
        """|disp| at the last step (the pool's episode stat; DuckEnv reads it from tracker.prev["disp"])."""
        return float(np.hypot(self._trk[5], self._trk[6]))


def _fast_worker(remote, cfg, seeds):
    """lab.pool._worker with FastDuckEnv (same protocol and outputs)."""
    import os
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(k, "1")
    envs = [FastDuckEnv(cfg["policy"], cfg["lo"], cfg["hi"], cfg["episode_s"], cfg["terminations"], s) for s in seeds]
    first = [e.reset() for e in envs]
    n = len(envs)
    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            remote.send((np.stack([o for o, _ in first]), np.stack([p for _, p in first])))
        elif cmd == "step":
            obs = np.empty((n, 61), np.float32)
            priv = np.empty((n, FastDuckEnv.PRIV), np.float32)
            S, term, tout, eps = np.empty((n, STATE_DIM), np.float32), np.zeros(n, bool), np.zeros(n, bool), []
            for i, (e, a) in enumerate(zip(envs, data)):
                o, p, s, tm, to = e.step(a)
                S[i], term[i], tout[i] = s, tm, to
                if tm or to:
                    eps.append(dict(len=e.k, fell=float(s[-3] > 0), terminated=float(tm), disp=e.disp_now))
                    o, p = e.reset()
                obs[i], priv[i] = o, p
            remote.send((obs, priv, S, term, tout, eps, np.array([e.k for e in envs])))
        elif cmd == "close":
            remote.close()
            return


class FastPool(Pool):
    """lab.pool.Pool with FastDuckEnv workers (same seeds per worker/env, same step/reset/close)."""

    def __init__(self, cfg: dict, workers: int = 8, per_worker: int = 16, seed: int = 0):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for w in range(workers):
            a, b = ctx.Pipe()
            seeds = [seed * 100_000 + w * 1000 + i for i in range(per_worker)]
            p = ctx.Process(target=_fast_worker, args=(b, cfg, seeds), daemon=True)
            p.start()
            self.remotes.append(a)
            self.procs.append(p)
        self.per, self.n = per_worker, workers * per_worker
