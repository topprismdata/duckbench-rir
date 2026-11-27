"""Wobbly Raft 1v1 (Challenge.id "wobbly-raft", PvP: a duel Challenge; v2 since LB-106 plays the storm raft, v3 adds the anti-sit
rule DuelRules.low_out, judged in metrics.py MatchPoints): state columns of the
duel training env (duck_rl/duel_env.py DuelEnv), from the learner's side. Ported from the 2026-10-15 night (kit 1.9.0,
LB-108).

The match (info duel.rules; judge duck_rl/metrics.py MatchPoints, out "fell_off" at platform.height): a round raft
on a ball joint under its centre tips toward the weight on it (a soft spring pulls it level, the water damps it); from
the raft's swell_from_s a swell torque rises over swell_ramp_s to swell_nm and turns round the raft every
swell_period_s (duck_rl/duel.py `swell`, match.ts applySwell; v2: 7 N m from 4 s, full at 14 s). Out = the trunk centre
dropped into the water, or (v3) its trunk centre stayed under 9 cm above the raft's top for 4 s ("too_low"); the
last duck aboard wins; both aboard at the cap: both lose (`lost` = 1, 0 points). Columns:
Sumo's (challenges/sumo.py; d_self / d_opp = each trunk's xy distance from the raft centre, opp_out_speed = the rival's
radial speed outward, margin_* against the raft's radius) plus raft_tilt. The critic's extras are Sumo's.
"""
from __future__ import annotations

import math

from .sumo import SumoAdapter


class WobblyRaftAdapter(SumoAdapter):
    COLUMNS = SumoAdapter.COLUMNS + ["raft_tilt"]
    DOC = {**SumoAdapter.DOC, "raft_tilt": "the raft's tilt from level (deg)"}

    def __init__(self, env):
        super().__init__(env)
        import mujoco
        m = env.world.m
        raft = next((p for p in self.info["duel"].get("scene") or [] if p["kind"] == "raft"), None)
        if raft is None:
            raise ValueError("wobbly-raft: no raft prop in the info's duel.scene (re-export it with ts/prepare.ts)")
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{raft['id']}_ball")
        self.q = int(m.jnt_qposadr[j])

    def fill(self, st, EX, row) -> None:
        super().fill(st, EX, row)
        _w, x, y, _z = self.env.world.d.qpos[self.q:self.q + 4]  # the ball joint's quaternion: tilt = the z axis' angle
        st[EX["raft_tilt"]] = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))
