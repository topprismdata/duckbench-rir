"""Bread Fight's crumbs in Python (kit 1.10.0): a line-for-line twin of packages/sim/src/runtime/crumbs.ts createCrumbs
for a still crumb field (DuelRules.bread without flee / cells / goose / brood: Laser Dot, Duck-Man and Mama's Brood are
refused by duel.py check_ported), so a Python match draws the same crumb spots from the shared ":match" stream.

  spots     `out` crumbs (BreadRules.out, absent = 1), props crumb_ids(rules) ("crumb", "crumb2", "crumb3"), all due at
            t = 0; an eaten one is due again gap_s later;
  update    on the state of row k (t = k / ticks_per_s), called by the judge (metrics.py MatchPoints.push) on every row
            until the decision, row 0 included (a missed or extra call shifts every later draw): for crumb j = 0..k-1 in
            order, if it is not out, it is due (t + 1e-9 >= due) and nobody has to_win: `candidates` times r = area_r *
            sqrt(U(0, 1)), then a = U(-pi, pi); the spot = the first candidate with the lowest cost, cost = max(d) -
            min(d) over the ducks' trunk distances d, + 10 + (min_dist - min(d)) when min(d) < min_dist, + for each other
            crumb out (index order, those placed earlier this row included) within sep: 10 + (sep - its distance).
            Then one eat (none once a duck has to_win): the nearest (duck, crumb) pair in reach (trunk centre or head
            within eat_r; a duck down eats nothing), ties to the lower crumb, then the lower slot. Returns the eater's
            slot or -1;
  poses     each crumb prop's free-joint pose at time t (prop-drive.ts: written right before the tick of row k with t =
            (k + 1) / ticks_per_s, and at the reset with t = 0): at its spot while out, a POP_S hop after it is eaten
            (out > 1), else HIDDEN under the floor. The props are weightless (gravcomp) and collide with nothing, so the
            pose only shows in qpos (the trajectory, the replay).
JS math: Math.hypot -> jsmath.js_hypot2, Math.sqrt -> math.sqrt (exact); Math.cos / Math.sin / Math.atan2 -> math's
(macOS libm vs V8's fdlibm may differ by 1 ulp, as the spawn rotation; that can only flip a near-tie, which
tools/duel_parity.py's logic replay would show).
"""
from __future__ import annotations

import math

from .jsmath import js_hypot2

HIDDEN = (0.0, 0.0, -0.5, 1.0, 0.0, 0.0, 0.0)  # crumbs.ts HIDDEN: the crumb body under the floor
POP_S = 0.4  # the eaten hop (out > 1 only): s (crumbs.ts POP_S; 0.25 s / 9 cm until kit 1.10.1)
POP_H = 0.18  # m
REFUSED = ("flee", "cells", "goose", "brood")  # BreadRules variants the Python field does not run


def crumb_ids(rules: dict) -> list[str]:
    """schema breadCrumbIds: `crumb`, then `<crumb>2`, `<crumb>3`, ... (one prop per crumb out at once)."""
    return [rules["crumb"] if i == 0 else f"{rules['crumb']}{i + 1}" for i in range(int(rules.get("out") or 1))]


class CrumbField:
    """crumbs.ts createCrumbs (module doc). counts[slot]; spots[j] (x, y) or None; update(t, ducks) with ducks =
    [(x, y, hx, hy, down)] by slot; pose(j, t); nearest / second (x, y) -> the spot or None; current()."""

    def __init__(self, rules: dict, n: int, centre, prng):
        bad = [k for k in REFUSED if rules.get(k) is not None]
        if bad:
            raise ValueError(f"BreadRules {', '.join(bad)}: not in the Python crumb field (Laser Dot, Duck-Man, Mama's Brood)")
        self.rules, self.prng = rules, prng
        self.cx, self.cy = float(centre[0]), float(centre[1])
        self.to_win, self.eat_r, self.gap_s = int(rules["to_win"]), float(rules["eat_r"]), float(rules["gap_s"])
        self.area_r, self.min_dist = float(rules["area_r"]), float(rules["min_dist"])
        self.candidates = int(rules["candidates"])
        self.sep = None if rules.get("sep") is None else float(rules["sep"])
        self.ids = crumb_ids(rules)
        k = self.k = len(self.ids)
        self.counts = [0] * n
        self.spots: list[tuple[float, float] | None] = [None] * k
        self.due = [0.0] * k
        self.eaten_at: list[tuple[float, float] | None] = [None] * k
        self.eaten_t = [-1.0] * k

    def _place(self, ducks, j: int) -> tuple[float, float]:
        best, best_c = (self.cx, self.cy), math.inf
        u = self.prng.uniform
        for _ in range(self.candidates):
            r = self.area_r * math.sqrt(u(0.0, 1.0))
            a = u(-math.pi, math.pi)
            p0, p1 = self.cx + r * math.cos(a), self.cy + r * math.sin(a)
            d = [js_hypot2(q[0] - p0, q[1] - p1) for q in ducks]
            lo = min(d)
            cost = max(d) - lo + ((10 + (self.min_dist - lo)) if lo < self.min_dist else 0)
            if self.sep is not None:
                for m in range(self.k):
                    o = self.spots[m]
                    if m == j or o is None:
                        continue
                    dm = js_hypot2(o[0] - p0, o[1] - p1)
                    if dm < self.sep:
                        cost += 10 + (self.sep - dm)
            if cost < best_c:
                best_c, best = cost, (p0, p1)
        return best

    def _near(self, x: float, y: float, skip: int) -> int:
        best, bd = -1, math.inf
        for j in range(self.k):
            s = self.spots[j]
            if j == skip or s is None:
                continue
            d = js_hypot2(s[0] - x, s[1] - y)
            if d < bd:
                bd, best = d, j
        return best

    def current(self):
        """The lowest-numbered crumb out, or None."""
        return next((s for s in self.spots if s is not None), None)

    def nearest(self, x: float, y: float):
        j = self._near(x, y, -1)
        return None if j < 0 else self.spots[j]

    def second(self, x: float, y: float):
        j = self._near(x, y, self._near(x, y, -1))
        return None if j < 0 else self.spots[j]

    def update(self, t: float, ducks) -> int:
        """Row time t, ducks [(x, y, hx, hy, down)] by slot (module doc). Returns the eater's slot or -1."""
        won = any(c >= self.to_win for c in self.counts)
        for j in range(self.k):
            if self.spots[j] is not None or t + 1e-9 < self.due[j] or won:
                continue
            self.spots[j] = self._place(ducks, j)
        eater, eaten, best_d = -1, -1, math.inf
        if not won:
            for j in range(self.k):
                s = self.spots[j]
                if s is None:
                    continue
                for i, q in enumerate(ducks):
                    if q[4]:
                        continue
                    d = min(js_hypot2(q[0] - s[0], q[1] - s[1]), js_hypot2(q[2] - s[0], q[3] - s[1]))
                    if d <= self.eat_r and d < best_d:
                        best_d, eater, eaten = d, i, j
        if eater >= 0:
            self.counts[eater] += 1
            self.eaten_at[eaten] = self.spots[eaten]
            self.eaten_t[eaten] = t
            self.spots[eaten] = None
            self.due[eaten] = t + self.gap_s
        return eater

    def pose(self, j: int, t: float) -> tuple:
        """crumbs.ts drives[j].at(t): the prop's free-joint qpos (x y z, quat w x y z)."""
        s = self.spots[j]
        if s is not None:
            return (s[0], s[1], 0.0, 1.0, 0.0, 0.0, 0.0)
        e = self.eaten_at[j]
        u = (t - self.eaten_t[j]) / POP_S
        if self.k == 1 or e is None or u < 0 or u >= 1:
            return HIDDEN
        yaw = (math.pi / 2) * u  # the quaternion's half angle: half a turn over the pop (crumbs.ts)
        return (e[0], e[1], POP_H * math.sin((math.pi / 2) * u), math.cos(yaw), 0.0, 0.0, math.sin(yaw))


class CrumbSense:
    """match.ts's SensorOptions.crumbs for slot `slot` of a 1v1 match: the crumb field and the other duck's trunk xy as
    the judge holds it on the current row (MatchPoints x / y)."""

    def __init__(self, field: CrumbField, judge, slot: int):
        self.f, self.j, self.slot = field, judge, slot

    def at(self, x: float, y: float):
        c = self.f.nearest(x, y)
        return c if c is not None else self.f.current()

    def second(self, x: float, y: float):
        return self.f.second(x, y)

    def rival_dist_to(self, c) -> float:
        o = 1 - self.slot
        return js_hypot2(self.j.x[o] - c[0], self.j.y[o] - c[1])

    def mine(self) -> int:
        return self.f.counts[self.slot]

    def rivals(self) -> int:
        return max(0, self.f.counts[1 - self.slot])
