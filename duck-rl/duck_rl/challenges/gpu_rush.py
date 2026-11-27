"""GPU Rush 1v1 (Challenge.id "gpu-rush-1v1" v2, was Bread Fight 1v1 `bread-fight`; PvP: a duel Challenge; kit 1.10.0, renamed
in 1.10.1; the engine and these columns keep the crumb words: a crumb is a GPU, eaten = grabbed): state columns and critic extras of the duel
training env (duck_rl/duel_env.py DuelEnv), from the learner's side ("self" = the learner's duck, "opp" = the other).

The match (info duel.rules.bread; the field duck_rl/crumbs.py, the judge duck_rl/metrics.py MatchPoints): three crumbs
lie within 0.75 m of the bank's centre; the first duck whose trunk centre or head comes within 12 cm of one eats it, and
a new one appears 0.6 s later at a spot about as far from both ducks and 40 cm from the other crumbs. The first duck to
eat 5 wins ("ate_three"); at 30 s more crumbs wins ("more_crumbs"), level = both lose ("timeout_both_lose": `lost`).
Shoving is allowed; a fall loses nothing but time (no down rule; the ring at 3 m is never reached in practice).

Columns: Sumo's (challenges/sumo.py: events, distances from the bank's centre, gap, tilts, ...) plus
  crumbs_self / crumbs_opp   crumbs eaten so far;
  ate_self / ate_opp         1 on the row that duck ate a crumb;
  crumb_dist / crumb_bearing the nearest crumb out to the learner's trunk (m; rad from its heading, + = left), 5 / 0
                             when none is out (the duckscript's crumb_dist / crumb_bearing, exact, no noise);
  crumb_rival_dist           the other duck's trunk distance to that crumb (m; 5 when none);
  crumb2_dist / crumb2_bearing / crumb2_rival_dist  the same for the second-nearest crumb;
  crumb_close / crumb2_close the closing speed to the nearest / second-nearest crumb (m/s, + = getting closer): the drop
                             of that distance since the last row / 0.02 s; 0 on the episode's first row and on any row
                             where the set of crumbs out changed (one appeared or was eaten).
Critic priv (6, the learner's fixed batch layout): [nearest crumb dist, its bearing, the rival's dist to it, the second
crumb's dist, (crumbs_self - crumbs_opp) / to_win, t / the Challenge's duration].
"""
from __future__ import annotations

import math

import numpy as np

from .sumo import SumoAdapter
from ..jsmath import js_hypot2

NONE_D = 5.0  # sensors.ts: a distance with no crumb out


class GpuRushAdapter(SumoAdapter):
    COLUMNS = SumoAdapter.COLUMNS + ["crumbs_self", "crumbs_opp", "ate_self", "ate_opp", "crumb_dist", "crumb_bearing",
                                     "crumb_rival_dist", "crumb2_dist", "crumb2_bearing", "crumb2_rival_dist",
                                     "crumb_close", "crumb2_close"]
    DOC = {**SumoAdapter.DOC,
           "crumbs_self, crumbs_opp": "crumbs eaten so far by the learner / the other duck (the first to 5 wins)",
           "ate_self, ate_opp": "1 on the row that duck ate a crumb",
           "crumb_dist, crumb_bearing": "the nearest crumb out to the learner's trunk: m, and rad from its heading "
                                        "(+ = left); 5 and 0 when none is out",
           "crumb_rival_dist": "the other duck's trunk distance to the learner's nearest crumb (m; 5 when none)",
           "crumb2_dist, crumb2_bearing, crumb2_rival_dist": "the same for the second-nearest crumb",
           "crumb_close, crumb2_close": "closing speed to the nearest / second-nearest crumb (m/s, + = closer); 0 on "
                                        "the first row and on rows where a crumb appeared or was eaten"}

    def __init__(self, env):
        super().__init__(env)
        b = self.info["duel"]["rules"].get("bread")
        if b is None:
            raise ValueError(f"{self.info['challenge']['id']}: no duel.rules.bread in the info (not a GPU Rush challenge)")
        self.to_win = float(b["to_win"])
        self._prev = None  # (the spots tuple, d1, d2) of the last row

    def on_reset(self, row) -> None:
        self._prev = None

    def _crumbs(self, row):
        """((d1, b1, r1), (d2, b2, r2)) for the learner from DuelRow `row` (crumbs.py nearest / second; sensors.ts)."""
        w = self.env.world
        f = w.crumbs
        k = w.ducks[w.slot_of[0]]
        q = w.d.qpos
        x, y = float(q[k.root_qpos_adr]), float(q[k.root_qpos_adr + 1])
        o = w.ducks[w.slot_of[1]]
        ox, oy = float(q[o.root_qpos_adr]), float(q[o.root_qpos_adr + 1])
        xm = w.d.xmat[k.trunk]
        hdg = math.atan2(xm[3], xm[0])
        out = []
        for c in (f.nearest(x, y), f.second(x, y)):
            if c is None:
                out.append((NONE_D, 0.0, NONE_D))
            else:
                b = math.atan2(c[1] - y, c[0] - x) - hdg
                out.append((js_hypot2(c[0] - x, c[1] - y), (b + math.pi) % (2 * math.pi) - math.pi,
                            js_hypot2(ox - c[0], oy - c[1])))
        return out

    def fill(self, st, EX, row) -> None:
        super().fill(st, EX, row)
        st[EX["crumbs_self"]], st[EX["crumbs_opp"]] = row.crumbs[0], row.crumbs[1]
        st[EX["ate_self"]], st[EX["ate_opp"]] = float(row.ate[0]), float(row.ate[1])
        (d1, b1, r1), (d2, b2, r2) = self._crumbs(row)
        st[EX["crumb_dist"]], st[EX["crumb_bearing"]], st[EX["crumb_rival_dist"]] = d1, b1, r1
        st[EX["crumb2_dist"]], st[EX["crumb2_bearing"]], st[EX["crumb2_rival_dist"]] = d2, b2, r2
        spots = tuple(row.spots)
        p = self._prev
        same = p is not None and p[0] == spots
        st[EX["crumb_close"]] = (p[1] - d1) / 0.02 if same and d1 < NONE_D else 0.0
        st[EX["crumb2_close"]] = (p[2] - d2) / 0.02 if same and d2 < NONE_D else 0.0
        self._prev = (spots, d1, d2)

    def priv(self, row, out: np.ndarray) -> None:
        (d1, b1, r1), (d2, _b2, _r2) = self._crumbs(row)
        out[0], out[1], out[2], out[3] = d1, b1, r1, d2
        out[4] = (row.crumbs[0] - row.crumbs[1]) / self.to_win
        out[5] = row.t / self.challenge_s
