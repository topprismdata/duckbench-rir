"""Sumo (Challenge.id "sumo", PvP: a duel Challenge): the state columns and the critic extras of the duel training env
(duck_rl/duel_env.py DuelEnv), from the learner's side: "self" = the learner's duck (side A), "opp" = the opponent's
(side B), whichever slot each plays this episode.

The match (info duel.rules): two ducks on a round stage (top at platform.height); a duck whose trunk drops
fell_off_drop_m below the top has fallen off and loses; from the shrink's from_s (15 s) on, the safe circle shrinks
from ring.radius (0.5 m) to radius_end (0.3 m) at 20 s and a trunk outside it loses ("red_zone"); a duck down (tilt
> 60 deg for 0.5 s, until upright again) for more than down_loses_s (3 s) loses; at 20 s the trunk closer to the ring
centre wins (within draw_band_m a draw). The judge is duck_rl/metrics.py MatchPoints; DuelRow carries its per-row state.

Event columns are 1 on the episode's last row only (the env ends the episode there): the decision row (won / lost /
draw / invalid, decided, points) or the row the learner left its trained nodes (unbound). A reward pays an event once
by dividing by the tick (x / 0.02, as the solo configs' one-off terms). Critic priv: [2 vx, 2 vy (the learner's trunk
velocity in its heading frame, the solo env's priv[0:2]), d_self, d_opp, gap, t / the Challenge's duration].
"""
from __future__ import annotations

import math

import numpy as np

from . import GenericAdapter


class SumoAdapter(GenericAdapter):
    COLUMNS = ["x", "y", "hdg", "won", "lost", "draw", "invalid", "decided", "points", "unbound", "d_self", "d_opp",
               "ring_r", "margin_self", "margin_opp", "out_speed_self", "opp_out_speed", "gap", "opp_bearing",
               "upright_self", "upright_opp", "tilt_self", "tilt_opp", "down_self", "down_opp", "z_self", "z_opp",
               "touching", "node_i"]
    DOC = {
        "x, y": "the learner's trunk position (m, world; the ring centre is info duel.rules.ring x, y: (0, 0))",
        "hdg": "the learner's heading (rad, world)",
        "won, lost, draw": "1 on the decision row by its outcome for the learner (a draw: both out or down on one row, "
                           "or the 20-s time-out within draw_band_m; a time-out that both lose is `lost`)",
        "invalid": "1 on the decision row of an invalid match (the physics broke: no points to either); the row's "
                   "other columns repeat the episode's last valid row's (duck_rl/duel_env.py: no NaN from the broken "
                   "state)",
        "decided": "1 on the decision row (any outcome); the episode ends there",
        "points": "the learner's points on the decision row (1 win, 0.5 draw, 0 loss / invalid), else 0",
        "unbound": "1 on the row the learner's active node is not a trained node (e.g. recover: get_up after a "
                   "fall); the episode ends there (UNBOUND 'end')",
        "d_self, d_opp": "each trunk's xy distance from the ring centre (m)",
        "ring_r": "the safe circle's radius this row (m: 0.5, shrinking to 0.3 over 15-20 s)",
        "margin_self, margin_opp": "ring_r - d (m; < 0 = outside the safe circle, a loss from 15 s on)",
        "out_speed_self, opp_out_speed": "each trunk's radial speed away from the ring centre (m/s, + = outward)",
        "gap": "the xy distance between the two trunks (m; ~0.15-0.2 when they touch)",
        "opp_bearing": "the direction to the opponent's trunk from the learner's heading (rad, + = left, in [-pi, pi])",
        "upright_self, upright_opp": "1 while the trunk tilt is under the judge's upright threshold (32 deg)",
        "tilt_self, tilt_opp": "trunk tilt (deg)",
        "down_self, down_opp": "the judge's raw fall state: 1 from tilt > 60 deg for 0.5 s until upright again (down "
                               "for > 3 s loses)",
        "z_self, z_opp": "trunk height above the stage top (m; 0.05 below it is off the stage); use z_self, not the "
                         "lab columns height, head_z and feet_z: those are measured from the floor, and the stage top "
                         "is info duel.rules.platform.height (0.625 m) above it",
        "touching": "1 while the ducks touch (any contact between a geom of one and a geom of the other)",
        "node_i": "the learner's active node: its index in the duckscript's node order (the compiled program's key "
                  "order), -1 = END",
    }

    def __init__(self, env):
        super().__init__(env)
        ring = self.info["duel"]["rules"]["ring"]
        self.cx, self.cy = float(ring["x"]), float(ring["y"])
        self.challenge_s = float(self.info["duration_s"])

    def _trunks(self):
        """(learner xy, opponent xy, learner xy velocity, opponent xy velocity) in the world frame (a free joint's
        qvel[0:3] is the world-frame linear velocity)."""
        w = self.env.world
        q, v = w.d.qpos, w.d.qvel
        out = []
        for side in (0, 1):
            k = w.ducks[w.slot_of[side]]
            out.append((q[k.root_qpos_adr], q[k.root_qpos_adr + 1], v[k.root_dof_adr], v[k.root_dof_adr + 1]))
        return out

    def _radial(self, x: float, y: float, vx: float, vy: float) -> float:
        dx, dy = x - self.cx, y - self.cy
        r = math.hypot(dx, dy)
        return (dx * vx + dy * vy) / r if r > 1e-9 else 0.0

    def fill(self, st, EX, row) -> None:
        """The columns of DuelRow `row` (the state after the learner's tick) into the state row st."""
        env = self.env
        (xs, ys, vxs, vys), (xo, yo, vxo, vyo) = self._trunks()
        w = env.world
        xm = w.d.xmat[w.ducks[w.slot_of[0]].trunk]
        hdg = math.atan2(-xm[1], xm[4])
        st[EX["x"]], st[EX["y"]], st[EX["hdg"]] = xs, ys, hdg
        self.fill_events(st, EX, row)
        st[EX["d_self"]], st[EX["d_opp"]] = row.dist[0], row.dist[1]
        st[EX["ring_r"]] = row.radius
        st[EX["margin_self"]], st[EX["margin_opp"]] = row.radius - row.dist[0], row.radius - row.dist[1]
        st[EX["out_speed_self"]] = self._radial(xs, ys, vxs, vys)
        st[EX["opp_out_speed"]] = self._radial(xo, yo, vxo, vyo)
        st[EX["gap"]] = math.hypot(xo - xs, yo - ys)
        b = math.atan2(yo - ys, xo - xs) - hdg
        st[EX["opp_bearing"]] = (b + math.pi) % (2 * math.pi) - math.pi
        st[EX["upright_self"]], st[EX["upright_opp"]] = float(row.upright[0]), float(row.upright[1])
        st[EX["tilt_self"]], st[EX["tilt_opp"]] = row.tilt[0], row.tilt[1]
        st[EX["down_self"]], st[EX["down_opp"]] = float(row.down[0]), float(row.down[1])
        st[EX["z_self"]], st[EX["z_opp"]] = row.z[0], row.z[1]
        st[EX["touching"]] = float(row.touching)
        st[EX["node_i"]] = float(row.node_index[0])

    def fill_events(self, st, EX, row) -> None:
        """The event columns of `row` (won, lost, draw, invalid, decided, points, unbound) into st: fill's, and the only
        columns DuelEnv.step writes on an invalid round's row (it reads nothing else of a broken state)."""
        decided, invalid = bool(row.decided), bool(row.invalid)
        st[EX["won"]] = float(decided and row.winner == 0)
        both_lose = row.reason == "timeout_both_lose"  # timeout "both_lose" (Wobbly Raft, Log Roll, Goose Chase): a loss
        st[EX["lost"]] = float(decided and (row.winner == 1 or both_lose))
        st[EX["draw"]] = float(decided and not invalid and row.winner is None and not both_lose)
        st[EX["invalid"]], st[EX["decided"]] = float(invalid), float(decided)
        st[EX["points"]] = float(row.points[0]) if decided else 0.0
        st[EX["unbound"]] = float(self.env.unbound_now)

    def priv(self, row, out: np.ndarray) -> None:
        """The critic's 6 extras for DuelRow `row` (module doc): what decides a match, which the policy cannot see."""
        (xs, ys, _, _), (xo, yo, _, _) = self._trunks()
        out[0], out[1] = row.priv[0, 0], row.priv[0, 1]
        out[2], out[3] = row.dist[0], row.dist[1]
        out[4] = math.hypot(xo - xs, yo - ys)
        out[5] = row.t / self.challenge_s
