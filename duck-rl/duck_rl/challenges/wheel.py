"""Hamster wheel: the wheel and where the duck is on its rim (from the wheel-rl kit's WheelEnv).

cfg: wheel_scale {mass, frictionloss, damping: [lo, hi]} (training-only bearing/mass randomisation), wheel_rpm0 [lo, hi]
(initial wheel speed, spawn starts). The episode runs to the Challenge's 15 s unless cfg end_s says otherwise. The metric
columns (window/best/dbest, aliases avg3_rpm/peak_rpm/dpeak_rpm) come from duck_rl/metrics.py WheelSpeed.
"""
from __future__ import annotations

import math

from . import HingedAdapter


class WheelAdapter(HingedAdapter):
    KIND = "wheel"
    SCALE_KEY, RPM0_KEY = "wheel_scale", "wheel_rpm0"
    COLUMNS = ["rim_speed", "rim_angle_deg"]
    ALIASES = {"wheel_omega": "prop_omega", "wheel_rpm": "prop_rpm", "wheel_turns": "prop_turns"}
    DOC = {"wheel_omega, wheel_rpm, wheel_turns": "= prop_omega (rad/s, + = the way a forward walk drives it), prop_rpm, prop_turns",
           "rim_speed": "rim surface speed (m/s, signed)",
           "rim_angle_deg": "the duck's CoM angle up the rim from the bottom (deg, + = ahead of the axle)"}

    def __init__(self, env):
        super().__init__(env)
        m = env.duck.model
        self.radius = float(self.prop["geometry"]["radius"])
        self.axle_x = float(m.body_pos[self.body][0])

    def fill(self, st, EX, d) -> None:
        super().fill(st, EX, d)
        com = d.subtree_com[self.env.duck.trunk]
        st[EX["rim_speed"]] = float(d.qvel[self.v]) * self.radius
        st[EX["rim_angle_deg"]] = math.degrees(math.asin(max(-1.0, min(1.0, (com[0] - self.axle_x) / self.radius))))
