"""Lazy Susan (turntable): the disc and the duck's orbit round its axle (from the disc-rl kit's DiscEnv).

cfg: disc_scale {mass, frictionloss, damping: [lo, hi]} (training-only bearing/mass randomisation), disc_rpm0 [lo, hi]
(signed initial disc speed, spawn starts). The episode ends at the bell (at_s) unless cfg end_s says otherwise: nothing
after the bell counts (the Challenge's last 5 s are only for the clip).
"""
from __future__ import annotations

import math

import mujoco

from . import HingedAdapter


class TurntableAdapter(HingedAdapter):
    KIND = "turntable"
    SCALE_KEY, RPM0_KEY = "disc_scale", "disc_rpm0"
    COLUMNS = ["r", "phi", "v_tan_rel", "v_rad", "L_duck", "L_disc", "yaw_rate_w"]
    ALIASES = {"disc_omega": "prop_omega", "disc_rpm": "prop_rpm", "disc_angle": "prop_angle"}
    DOC = {"disc_omega, disc_rpm, disc_angle": "= prop_omega (rad/s, + = counter-clockwise from above), prop_rpm, prop_angle",
           "r, phi": "trunk distance from the disc axis (m) and its angle (rad)",
           "v_tan_rel": "trunk tangential speed relative to the disc surface under it (m/s, + = counter-clockwise)",
           "v_rad": "trunk radial speed (m/s, + = outwards)",
           "L_duck": "duck angular momentum about the disc axis (kg m^2/s, + = counter-clockwise)",
           "L_disc": "disc angular momentum (kg m^2/s)", "yaw_rate_w": "trunk yaw rate in the world (rad/s)"}

    def __init__(self, env):
        super().__init__(env)
        self.cx, self.cy = float(self.prop["x"]), float(self.prop["y"])

    def default_end_s(self) -> float:
        return float(self.info["metric"]["params"]["at_s"])

    def fill(self, st, EX, d) -> None:
        super().fill(st, EX, d)
        m, root = self.env.duck.model, self.env.duck_root
        w = float(d.qvel[self.v])
        x, y = float(d.qpos[0]) - self.cx, float(d.qpos[1]) - self.cy
        vx, vy = float(d.qvel[0]), float(d.qvel[1])
        r = math.hypot(x, y)
        ex, ey = (x / r, y / r) if r > 1e-6 else (1.0, 0.0)
        v_tan = -ey * vx + ex * vy
        mujoco.mj_subtreeVel(m, d)
        mc = m.body_subtreemass[root]
        com, lv = d.subtree_com[root], d.subtree_linvel[root]
        st[EX["r"]], st[EX["phi"]] = r, math.atan2(y, x)
        st[EX["v_tan_rel"]], st[EX["v_rad"]] = v_tan - w * r, ex * vx + ey * vy
        st[EX["L_duck"]] = mc * ((com[0] - self.cx) * lv[1] - (com[1] - self.cy) * lv[0]) + d.subtree_angmom[root][2]
        st[EX["L_disc"]] = m.body_inertia[self.body][2] * w
        st[EX["yaw_rate_w"]] = d.qvel[5]
