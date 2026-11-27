"""Saucer (Challenge.id "saucer": the real packages/challenges Saucer v1, and the spike Challenges saucer-spike / saucer-peak of
ts/spike_challenges.ts): the Lazy Susan disc with its axle tilted by tilt_deg toward the prop's +x (the low side: world -x on
the real Challenge, whose disc has yaw pi; world +x on the spike's, yaw 0), driven by the duck's weight. The duck starts at the
side (phi 90 deg from the low point), facing uphill along the tangent. Score: the real Challenge disc_peak_rpm over 0-10 s of
15 s (metrics.py DiscPeakRpm, until_s); the spikes disc_turns / disc_peak_rpm over 10 s. Every position and velocity is
taken in the prop's frame (turned by its yaw), so the columns mean the same on both.

cfg: disc_mass_log [lo, hi] kg: training-only disc mass, log-uniform per episode, the inertia scaled with it (a uniform
disc at this radius); disc_scale / disc_rpm0 as the turntable's. Critic: priv[2:4] = the disc-plane position (x', y'). Columns are in the disc plane (x' toward the low
side, y' toward the start side, origin on the top face's centre); the episode ends at the metric's until_s (the real
Challenge: the 10-s bell; nothing later counts), else at the Challenge's duration (cfg end_s overrides).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .turntable import TurntableAdapter


class SaucerAdapter(TurntableAdapter):
    COLUMNS = ["r", "phi", "v_tan_rel", "v_rad", "L_duck", "L_disc", "yaw_rate_w", "cw", "hdg_tan", "v_rel", "v_against",
               "tread", "tread05", "tread_peak", "dtread", "hdist", "fling", "dfling", "fling1", "dfling1", "disc_mass",
               "disc_R", "rpm_w05", "rpm_f", "throw20", "dthrow20"]
    DOC = dict(TurntableAdapter.DOC, **{
        "r, phi": "trunk distance from the axle in the disc plane (m) and its angle from the LOW point (rad, + toward +y, "
                  "the start side is +pi/2)",
        "v_tan_rel": "trunk tangential speed relative to the disc surface under it (m/s, + = counter-clockwise)",
        "cw": "the disc's clockwise speed (rad/s): the way the duck's weight drives it from the start side",
        "v_rel": "|trunk velocity - the disc surface's velocity under it| in the disc plane (m/s): running speed over the surface",
        "v_against": "its tangential part against the surface motion (m/s, + = running against the disc's spin)",
        "hdg_tan": "cos of the angle between the duck's heading and +e_phi (counter-clockwise tangent): 1 = facing against "
                   "a clockwise-turning surface (the start heading, uphill at the side; +y at the low point)",
        # targets (experiment 2026-09-30-saucer-targets-rl; mirrors of ts/challenges/saucer.ts treadmill / fling_dist)
        "tread": "the disc surface speed under the trunk this tick, |angle change| / dt x r (m/s)",
        "tread05": "its mean over the last 0.5 s (m/s; 0 before a full window)",
        "tread_peak": "the treadmill target so far: the best tread05 until the first floor contact (frozen after it)",
        "dtread": "what this tick added to tread_peak (m/s)",
        "hdist": "the trunk's horizontal distance from the disc axle at the trunk's height (m)",
        "fling": "the fling-distance target so far: the max hdist since the first floor contact (0 before it)",
        "dfling": "what this tick added to fling (m)",
        "fling1": "the throw target (E): the max hdist over the first 1 s after the first floor contact (0 before it)",
        "dfling1": "what this tick added to fling1 (m)",
        "disc_mass": "this episode's disc mass (kg; training randomisation)",
        # 2026-10-01 (saucer treadmill / throw series)
        "disc_R": "this env's disc radius (m; train.py WORLDS gives envs different diameters)",
        "rpm_w05": "the disc's 0.5-s mean |rpm| (0 before a full window; ts/challenges/saucer.ts rpmAt; = the peak mirror's rpm05)",
        "rpm_f": "rpm_w05 at the first floor contact's row (0 before it): throw20's gate",
        "throw20": "the throw target: fling1 if rpm_f >= 20 else 0 (m; ts/challenges/saucer.ts throw20)",
        "dthrow20": "what this tick added to throw20 (m)",
    })
    TREAD_ROWS = 25  # 0.5 s at 50 Hz

    def __init__(self, env):
        super().__init__(env)
        a = math.radians(float(self.prop["params"].get("tilt_deg", 0.0)))
        self.ca, self.sa = math.cos(a), math.sin(a)
        yaw = float(self.prop.get("yaw") or 0.0)
        self.cy, self.sy = math.cos(yaw), math.sin(yaw)  # the prop frame: x' = dx cy + dy sy, y' = -dx sy + dy cy
        h = float(self.prop["geometry"]["top"])
        # the top-face centre: the body origin (the compiled pose) + h along the axle (sin a cy, sin a sy, cos a)
        body_pos = env.duck.model.body_pos[self.body]
        self.c = (float(body_pos[0]) + h * self.sa * self.cy, float(body_pos[1]) + h * self.sa * self.sy, float(body_pos[2]) + h * self.ca)
        self.o = (float(body_pos[0]), float(body_pos[1]), float(body_pos[2]))  # the axle's point at the body origin
        self.ta = self.sa / self.ca
        self.R = float(self.prop["geometry"]["radius"])
        self._targets_reset()

    def _targets_reset(self) -> None:
        self._th_prev, self._win, self._tpeak, self._tstop, self._fling = None, deque(maxlen=self.TREAD_ROWS), 0.0, False, 0.0
        self._k_out, self._fling1 = None, 0.0
        self._th_hist, self._rpm_f, self._throw = deque(maxlen=self.TREAD_ROWS + 1), 0.0, 0.0

    def default_end_s(self) -> float:
        until = self.info["metric"]["params"].get("until_s")
        return float(until) if until is not None else float(self.info["duration_s"])

    def _rot(self, x: float, y: float) -> tuple[float, float]:
        """A horizontal world vector in the prop frame (x' toward the low side)."""
        return x * self.cy + y * self.sy, -x * self.sy + y * self.cy

    def priv_override(self, d, pr) -> None:
        """The critic sees the duck's place on the disc: priv[2] (2 vz) and priv[3] (10 (z - 0.12)) become 2 x' and 2 y'
        (the disc-plane position, m; y' is the gravity lever arm), the priv width stays 6 (the MLX learner's layout)."""
        dx, dy = self._rot(float(d.qpos[0]) - self.c[0], float(d.qpos[1]) - self.c[1])
        dz = float(d.qpos[2]) - self.c[2]
        pr[2] = 2.0 * (self.ca * dx - self.sa * dz)
        pr[3] = 2.0 * dy

    def reset(self, train: bool, spawn: bool) -> None:
        super().reset(train, spawn)
        self._targets_reset()
        rng_ = self.cfg.get("disc_mass_log")
        if train and rng_:
            lo, hi = rng_
            m_new = math.exp(self.env.rng.uniform(math.log(lo), math.log(hi)))
            m, p0 = self.env.duck.model, self._p0
            k = m_new / p0["mass"]
            m.body_mass[self.body] = m_new
            m.body_inertia[self.body] = p0["inertia"] * k

    def fill(self, st, EX, d) -> None:
        super().fill(st, EX, d)  # world-frame r/phi etc. first, overwritten below in the disc plane
        w = float(d.qvel[self.v])
        dx, dy = self._rot(float(d.qpos[0]) - self.c[0], float(d.qpos[1]) - self.c[1])
        dz = float(d.qpos[2]) - self.c[2]
        x, y = self.ca * dx - self.sa * dz, dy
        wvx, wvy = self._rot(float(d.qvel[0]), float(d.qvel[1]))
        vx, vy = self.ca * wvx - self.sa * float(d.qvel[2]), wvy
        r = math.hypot(x, y)
        ex, ey = (x / r, y / r) if r > 1e-6 else (1.0, 0.0)
        v_tan = -ey * vx + ex * vy
        st[EX["r"]], st[EX["phi"]] = r, math.atan2(y, x)
        st[EX["v_tan_rel"]], st[EX["v_rad"]] = v_tan - w * r, ex * vx + ey * vy
        st[EX["cw"]] = -w
        rvx, rvy = vx + w * y, vy - w * x  # minus the surface velocity w (-y, x)
        st[EX["v_rel"]] = math.hypot(rvx, rvy)
        st[EX["v_against"]] = -math.copysign(1.0, w) * (-ey * rvx + ex * rvy) if w != 0 else 0.0
        xm = d.xmat[self.env.duck.trunk]
        hx, hy = self._rot(float(xm[0]), float(xm[3]))  # trunk x axis (forward), projected, in the prop frame
        hn = math.hypot(hx, hy) or 1.0
        st[EX["hdg_tan"]] = (-hx * ey + hy * ex) / hn  # . e_phi = (-ey, ex)
        # targets (ts/challenges/saucer.ts treadmill / fling_dist): the per-tick angle change, so row k matches TS row k
        th = float(d.qpos[self.q])
        if self._th_prev is None:
            self._th_prev = float(self.env.duck.model.qpos0[self.q])
        tread = abs(th - self._th_prev) / 0.02 * r
        self._th_prev = th
        self._win.append(tread)
        t05 = sum(self._win) / self.TREAD_ROWS if len(self._win) == self.TREAD_ROWS else 0.0
        out = st[EX["out"]] > 0.5
        gain = 0.0
        if not self._tstop:
            if t05 > self._tpeak:
                gain, self._tpeak = t05 - self._tpeak, t05
            if out:
                self._tstop = True
        ox, oy = self._rot(float(d.qpos[0]) - self.o[0], float(d.qpos[1]) - self.o[1])
        hd = math.hypot(ox - (float(d.qpos[2]) - self.o[2]) * self.ta, oy)
        fgain = 0.0
        if out and hd > self._fling:
            fgain, self._fling = hd - self._fling, hd
        st[EX["tread"]], st[EX["tread05"]], st[EX["tread_peak"]], st[EX["dtread"]] = tread, t05, self._tpeak, gain
        st[EX["hdist"]], st[EX["fling"]], st[EX["dfling"]] = hd, self._fling, fgain
        g1 = 0.0
        if out and self._k_out is None:
            self._k_out = self.env.k  # the first row that sees the floor contact (TS outRow)
        if self._k_out is not None and self.env.k - self._k_out <= 50 and hd > self._fling1:  # rows outRow .. outRow + 1 s
            g1, self._fling1 = hd - self._fling1, hd
        st[EX["fling1"]], st[EX["dfling1"]] = self._fling1, g1
        st[EX["disc_mass"]] = float(self.env.duck.model.body_mass[self.body])
        st[EX["disc_R"]] = self.R
        self._th_hist.append(th)
        rpm05 = abs(th - self._th_hist[0]) / 0.5 * 60 / (2 * math.pi) if len(self._th_hist) == self.TREAD_ROWS + 1 else 0.0
        if out and self.env.k == self._k_out:
            self._rpm_f = rpm05
        g20 = 0.0
        if self._k_out is not None and self._rpm_f >= 20.0 and self._fling1 > self._throw:
            g20, self._throw = self._fling1 - self._throw, self._fling1
        st[EX["rpm_w05"]], st[EX["rpm_f"]], st[EX["throw20"]], st[EX["dthrow20"]] = rpm05, self._rpm_f, self._throw, g20
