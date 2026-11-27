"""Log Roll 1v1 (Challenge.id "log-roll", shipped as v2 in wave 2, and the prototype "log-roll-swift"; PvP duel Challenges):
state columns and critic extras of the duel training env (duck_rl/duel_env.py DuelEnv), from the learner's side. Ported
from the night (kit 1.9.0, LB-108). v2 (LB-109): from 15 s the current pushes the log round (the log's swell, duck_rl/duel.py).

The match (info duel.rules; judge duck_rl/metrics.py MatchPoints, win None, out "fell_off"): two ducks stand side by
side on top of a free-spinning log (a damped hinge about +x) facing +y; a duck is out when its trunk centre drops below
the log's axis (platform.height); the last one on wins; both on at the cap: both lose (`lost` = 1, 0 points). Columns:
Sumo's (challenges/sumo.py) plus log_off_self / log_off_opp (each trunk's y off the log's top line, m: the signal log_off),
opp_off_speed (d|log_off_opp|/dt, m/s, + = the rival slides away from the top line), log_spin (rad/s, + = the top moves
toward +y, the signal). Critic priv: [vx, vy (the learner's), log_off_self, log_off_opp, log_spin, t/T].
"""
from __future__ import annotations

import numpy as np

from .sumo import SumoAdapter


class LogRollAdapter(SumoAdapter):
    COLUMNS = SumoAdapter.COLUMNS + ["log_off_self", "log_off_opp", "opp_off_speed", "log_spin"]
    DOC = {**SumoAdapter.DOC,
           "log_off_self, log_off_opp": "each trunk's y minus the log's y (m; the log_off signal: + = toward +y)",
           "opp_off_speed": "how fast the rival leaves the top line: d|log_off_opp|/dt (m/s, + = away from it)",
           "log_spin": "the log's spin (rad/s; + = its top moves toward +y; the log_spin signal)"}

    def __init__(self, env):
        super().__init__(env)
        rt = env.world.rt
        if rt.log is None:
            raise ValueError("log-roll: no log prop in the info's duel.scene (re-export it with ts/prepare.ts)")
        import mujoco
        m = env.world.m
        self.log_y = rt.log[1]
        self.log_dof = int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{rt.log[0]}_hinge")])

    def fill(self, st, EX, row) -> None:
        super().fill(st, EX, row)
        (_xs, ys, _vxs, _vys), (_xo, yo, _vxo, vyo) = self._trunks()
        st[EX["log_off_self"]], st[EX["log_off_opp"]] = ys - self.log_y, yo - self.log_y
        st[EX["opp_off_speed"]] = float(np.sign(yo - self.log_y)) * vyo
        st[EX["log_spin"]] = -float(self.env.world.d.qvel[self.log_dof])

    def priv(self, row, out: np.ndarray) -> None:
        (_xs, ys, _, _), (_xo, yo, _, _) = self._trunks()
        out[0], out[1] = row.priv[0, 0], row.priv[0, 1]
        out[2], out[3] = ys - self.log_y, yo - self.log_y
        out[4] = -float(self.env.world.d.qvel[self.log_dof])
        out[5] = row.t / self.challenge_s
