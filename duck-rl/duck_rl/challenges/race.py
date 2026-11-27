"""The race duels (Challenge.duel.finish: Maze race "maze-race", Cliff "cliff"; PvP): Sumo's state columns and critic extras
(challenges/sumo.py) plus the race's own, from the learner's side. The judge (metrics.py MatchPoints) decides at the finish:
the first trunk within finish.radius of (finish.x, finish.y) and at least finish.z_min high wins at once, both on one
row is a draw ("both_finish"); at the time-out the trunk closer to the ring centre (the finish) wins, within
draw_band_m a draw; falls never lose (down_loses_s null).

Extra columns: d_fin_self / d_fin_opp (each trunk's xy distance to the finish, m), fin_rate_self (m/s the learner closes
on the finish this row, + = closer), lead (d_fin_opp - d_fin_self, m: + = ahead), z_up_self (the learner's trunk
height above the floor, m: Cliff's summit is 0.28 m up), climb_rate_self (m/s up), finished_self / finished_opp (1 on the decision row of a
finish by that side).
"""
from __future__ import annotations

import math

from .sumo import SumoAdapter


class RaceAdapter(SumoAdapter):
    COLUMNS = SumoAdapter.COLUMNS + ["d_fin_self", "d_fin_opp", "fin_rate_self", "lead", "z_up_self", "climb_rate_self",
                                     "finished_self", "finished_opp"]
    DOC = {**SumoAdapter.DOC,
           "d_fin_self, d_fin_opp": "each trunk's xy distance to the finish zone's centre (m; the zone is info duel.rules.finish)",
           "fin_rate_self": "how fast the learner closes on the finish (m/s, + = closer; from the last row)",
           "lead": "d_fin_opp - d_fin_self (m; + = the learner is ahead)",
           "z_up_self": "the learner's trunk height above the floor (m)",
           "climb_rate_self": "how fast the learner's trunk rises (m/s, from the last row; + = up)",
           "finished_self, finished_opp": "1 on the decision row when that side reached the finish (reason finish)"}

    def __init__(self, env):
        super().__init__(env)
        f = self.info["duel"]["rules"].get("finish")
        if f is None:
            raise ValueError(f"{self.info['challenge']['id']}: a race adapter needs duel.rules.finish")
        self.fx, self.fy = float(f["x"]), float(f["y"])
        self._prev = None
        self._prev_z = None

    def fill(self, st, EX, row) -> None:
        super().fill(st, EX, row)
        (xs, ys, _, _), (xo, yo, _, _) = self._trunks()
        ds, do = math.hypot(xs - self.fx, ys - self.fy), math.hypot(xo - self.fx, yo - self.fy)
        st[EX["d_fin_self"]], st[EX["d_fin_opp"]], st[EX["lead"]] = ds, do, do - ds
        st[EX["fin_rate_self"]] = 0.0 if self._prev is None or row.k == 0 else (self._prev - ds) / 0.02
        self._prev = None if row.stopped else ds
        w = self.env.world
        z = float(w.d.qpos[w.ducks[w.slot_of[0]].root_qpos_adr + 2])
        st[EX["z_up_self"]] = z
        st[EX["climb_rate_self"]] = 0.0 if self._prev_z is None or row.k == 0 else (z - self._prev_z) / 0.02
        self._prev_z = None if row.stopped else z
        fin = bool(row.decided) and row.reason == "finish"
        st[EX["finished_self"]] = float(fin and row.winner == 0)
        st[EX["finished_opp"]] = float(fin and row.winner == 1)
