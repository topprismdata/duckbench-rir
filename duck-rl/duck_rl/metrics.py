"""Python mirrors of the Challenge metrics, ONE per metric kind (packages/sim/src/runtime/metrics.ts), updated once per row
(the state before tick k = row k; row 0 = the reset state, pushed like TS does) with the same formula, so the training
env knows the Challenge score online. Parameters come from the Challenge info (ts/prepare.ts `metric.params`), never hard-coded.

A mirror: push(qpos, qvel) per row; `value` = the TS metric so far before the floor-out zeroing (the env applies
that); `settled` = the TS metric's settled() (success may be declared); restore(tail) continues from a bank state's
`metric_tail` (ts/metric_state.ts); columns() = its extra state columns (name -> value) and COLUMNS / ALIASES / DOC.
Kinds without a mirror get NoMirror (value NaN, never settled): the env still trains (reward from your own columns),
the in-loop TS eval is the score. To add a kind: a class here, registered in MIRRORS, and its tail in ts/metric_state.ts.
A duel Challenge's match_points (MatchPoints) is the match judge: one push per row with both ducks' postures (duck_rl/duel.py).
"""
from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from .jsmath import js_hypot2, js_round

RPM = 60.0 / (2.0 * math.pi)


def prop_of(info: dict, kind: str) -> dict:
    """The first hinged prop of this kind (info.props)."""
    for p in info.get("props", []):
        if p["kind"] == kind:
            return p
    raise ValueError(f"the challenge has no {kind} prop")


class NoMirror:
    kind = None
    COLUMNS: list[str] = []
    ALIASES: dict[str, str] = {}
    DOC: dict[str, str] = {}

    def __init__(self, info: dict, dt: float):
        self.kind = info["metric"]["kind"]
        self.row = 0

    def reset(self) -> None:
        self.row = 0

    def restore(self, tail: dict | None, row: int) -> None:
        self.row = row

    def push(self, qpos, qvel, **_kw) -> None:
        self.row += 1

    @property
    def value(self) -> float:
        return float("nan")

    @property
    def settled(self) -> bool:
        return False

    def columns(self) -> dict:
        return {}


class DiscSpeed(NoMirror):
    """disc_speed: |angle(row) - angle(row - back)| / (back dt) in rpm, back = min(row, n), n = window_s / dt; frozen
    from the bell row (at_s) on; settled once the bell row is in."""

    COLUMNS = ["live", "bell", "score"]
    ALIASES = {"live_rpm": "live"}
    DOC = {"live (live_rpm)": "|dtheta| over the metric window ending now, rpm (= the score if the bell rang now), never zeroed",
           "bell": "1 on the bell row (t == at_s)",
           "score": "the Challenge score from the bell row on (the metric, 0 if lost), else 0"}

    def __init__(self, info: dict, dt: float):
        super().__init__(info, dt)
        p = info["metric"]["params"]
        self.dt = dt
        self.at_s = float(p["at_s"])
        self.at_row = int(round(self.at_s / dt))
        self.n = max(1, int(round(float(p.get("window_s", 0.1)) / dt)))
        self.qadr = int(prop_of(info, "turntable")["qpos_adr"])
        self.reset()

    def reset(self) -> None:
        self.row = 0
        self.hist = deque(maxlen=self.n + 1)
        self.frozen = None
        self.live = 0.0

    def restore(self, tail: dict | None, row: int) -> None:
        self.reset()
        self.row = row
        if tail:
            for a in tail["hist"]:
                self.hist.append(float(a))
            self.frozen = tail.get("frozen")

    def push(self, qpos, qvel, **_kw) -> None:
        h = self.hist
        h.append(float(qpos[self.qadr]))
        self.live = abs(h[-1] - h[0]) / ((len(h) - 1) * self.dt) * RPM if len(h) > 1 else 0.0
        if self.row == self.at_row:
            self.frozen = self.live
        self.row += 1

    @property
    def value(self) -> float:
        return self.frozen if self.frozen is not None else self.live

    @property
    def settled(self) -> bool:
        return self.row > self.at_row

    def columns(self) -> dict:
        return dict(live=self.live, bell=float(self.row - 1 == self.at_row))


class WheelSpeed(NoMirror):
    """wheel_speed: the best full n-row window mean of |hinge speed| so far, rpm (n = window_s / dt, 3 s)."""

    COLUMNS = ["window", "best", "dbest"]
    ALIASES = {"avg3_rpm": "window", "peak_rpm": "best", "dpeak_rpm": "dbest"}
    DOC = {"window (avg3_rpm)": "the mean |wheel speed| over the last window (fewer rows at the start), rpm",
           "best (peak_rpm)": "the Challenge metric so far: the best full-window mean, rpm",
           "dbest (dpeak_rpm)": "what this row added to `best` (rpm; 0 unless a new best)"}

    def __init__(self, info: dict, dt: float):
        super().__init__(info, dt)
        self.n = int(round(float(info["metric"]["params"]["window_s"]) / dt))
        self.dof = int(prop_of(info, "wheel")["dof_adr"])
        self.reset()

    def reset(self) -> None:
        self.row = 0
        self.win = deque(maxlen=self.n)
        self.sum, self.best, self.gain = 0.0, 0.0, 0.0

    def restore(self, tail: dict | None, row: int) -> None:
        self.reset()
        self.row = row
        if tail:
            for w in tail["hist"]:
                self.push(0.0, None, w_abs=float(w))
            self.row = row
            self.best = max(self.best, float(tail.get("best", 0.0)))

    def push(self, qpos, qvel, w_abs: float | None = None, **_kw) -> None:
        w = abs(float(qvel[self.dof])) if w_abs is None else w_abs
        if len(self.win) == self.n:
            self.sum -= self.win[0]
        self.win.append(w)
        self.sum += w
        self.gain = 0.0
        if len(self.win) == self.n:
            v = self.sum / self.n * RPM
            if v > self.best:
                self.gain, self.best = v - self.best, v
        self.row += 1

    @property
    def value(self) -> float:
        return self.best

    @property
    def settled(self) -> bool:
        return True

    def columns(self) -> dict:
        return dict(window=self.sum / max(1, len(self.win)) * RPM, best=self.best, dbest=self.gain)


class WheelTurns(NoMirror):
    """wheel_turns (the Hamster wheel v2, a race): the most net turns so far, max |angle(row) - angle(row 0)| / 2 pi on the
    wheel's hinge (either direction; rocking does not add up). Always settled. The race's finish: the first row whose
    metric meets the Challenge's bar (metric_at_least: 1 turn; the TS `success` event), its t = row * dt, as MaxDistance."""

    COLUMNS = ["turns_live", "finished", "finish_now", "finish_t"]
    DOC = {"turns_live": "the wheel's signed turns from the start this row (+ = the way a forward run up the rim turns it)",
           "finished": "1 once the race is finished: the metric met the Challenge's bar (the TS `success` event)",
           "finish_now": "1 on the finishing row only",
           "finish_t": "the finish time (s, the sim's finishT) once finished, else the miss value (the Challenge's duration, "
                       "15 s: info ranking.miss_s, D-19); at the episode's end = the seed's race result"}
    MISS_S = 15.0  # the fallback; info ranking.miss_s wins

    def __init__(self, info: dict, dt: float):
        super().__init__(info, dt)
        self.dt = dt
        self.qadr = int(prop_of(info, "wheel")["qpos_adr"])
        self.MISS_S = float((info.get("ranking") or {}).get("miss_s") or info.get("duration_s") or self.MISS_S)
        s = info.get("finish") or info.get("success") or {}
        self.bar = float(s["metric_at_least"]) if "metric_at_least" in s else math.inf
        self.reset()

    def reset(self) -> None:
        self.row, self.a0, self.live, self.best, self.finish_t, self.now = 0, None, 0.0, 0.0, None, False

    def restore(self, tail: dict | None, row: int) -> None:
        self.reset()
        self.row = row
        if tail:
            if tail.get("kind") != "wheel_turns":
                raise ValueError(f"the bank's metric tail is {tail.get('kind')}, not wheel_turns: re-record the bank")
            self.a0 = float(tail["a0"])
            self.best = float(tail["best"])
            self.finish_t = None if tail.get("finish_t") is None else float(tail["finish_t"])

    def push(self, qpos, qvel, **_kw) -> None:
        a = float(qpos[self.qadr])
        if self.a0 is None:
            self.a0 = a
        self.live = (a - self.a0) / (2 * math.pi)
        self.best = max(self.best, abs(self.live))
        self.now = self.finish_t is None and self.best >= self.bar
        if self.now:
            self.finish_t = self.row * self.dt
        self.row += 1

    @property
    def value(self) -> float:
        return self.best

    @property
    def settled(self) -> bool:
        return True

    def columns(self) -> dict:
        return dict(turns_live=self.live, finished=float(self.finish_t is not None), finish_now=float(self.now),
                    finish_t=self.MISS_S if self.finish_t is None else self.finish_t)


class DiscTurns(NoMirror):
    """disc_turns (the Saucer spike): |angle(row) - angle(row 0)| / 2 pi, frozen on the first row that sees the duck on
    the floor (any substep since the last row) or fallen (tilt > 60 deg for 25 rows, row 0 included). The env passes
    `out` (floor touched so far) and `cos_tilt` (trunk z . world z) with every push; settled from the freeze on."""

    COLUMNS = ["turns_live", "stopped"]
    DOC = {"turns_live": "the disc's signed turns from the start (never frozen; + = counter-clockwise about the axle)",
           "stopped": "1 once the score froze (floor contact or on-disc fall)"}
    FALL_ROWS, COS_FALL = 25, 0.5

    def __init__(self, info: dict, dt: float):
        super().__init__(info, dt)
        self.qadr = int(prop_of(info, "turntable")["qpos_adr"])
        self.reset()

    def reset(self) -> None:
        self.row, self.a0, self.turns, self.live, self.stopped, self.run = 0, None, 0.0, 0.0, False, 0

    def restore(self, tail: dict | None, row: int) -> None:
        raise NotImplementedError("disc_turns has no bank tail: spawn starts only")

    def push(self, qpos, qvel, out: bool = False, cos_tilt: float = 1.0) -> None:
        a = float(qpos[self.qadr])
        if self.a0 is None:
            self.a0 = a
        self.live = (a - self.a0) / (2 * math.pi)
        self.run = self.run + 1 if cos_tilt < self.COS_FALL else 0
        if not self.stopped:
            self.turns = abs(self.live)
            if out or self.run >= self.FALL_ROWS:
                self.stopped = True
        self.row += 1

    @property
    def value(self) -> float:
        return self.turns

    @property
    def settled(self) -> bool:
        return self.stopped

    def columns(self) -> dict:
        return dict(turns_live=self.live, stopped=float(self.stopped))


class DiscPeakRpm(NoMirror):
    """disc_peak_rpm (Saucer; the spike's round 4): the best |angle(row) - angle(row - n)| / (n dt) in rpm over full windows
    (n = window_s / dt, 0.5 s), frozen on the first row that sees the duck on the floor (the env passes `out`; that row's
    window still counts) and after the row of until_s (the real Saucer: 10 s of a 15-s episode; none for the spike: never),
    as packages/sim/src/runtime/metrics.ts; a fall on the disc does not stop it. settled once frozen."""

    COLUMNS = ["rpm05", "peak", "dpeak", "stopped"]
    DOC = {"rpm05": "|disc speed| averaged over the metric window ending now (rpm; 0 before a full window), never frozen",
           "peak": "the Challenge metric so far: the best rpm05 until the first floor contact",
           "dpeak": "what this row added to `peak` (rpm; 0 unless a new best)",
           "stopped": "1 once the score froze (floor contact, or past until_s)"}

    def __init__(self, info: dict, dt: float):
        super().__init__(info, dt)
        self.dt = dt
        self.n = max(1, int(round(float(info["metric"]["params"].get("window_s", 0.5)) / dt)))
        until = info["metric"]["params"].get("until_s")
        self.last_row = None if until is None else int(round(float(until) / dt))  # rows 0 .. last_row count
        self.qadr = int(prop_of(info, "turntable")["qpos_adr"])
        self.reset()

    def reset(self) -> None:
        self.row, self.hist, self.best, self.gain, self.cur, self.stopped = 0, deque(maxlen=self.n + 1), 0.0, 0.0, 0.0, False

    def restore(self, tail: dict | None, row: int) -> None:
        raise NotImplementedError("disc_peak_rpm has no bank tail: spawn starts only")

    def push(self, qpos, qvel, out: bool = False, cos_tilt: float = 1.0) -> None:
        h = self.hist
        h.append(float(qpos[self.qadr]))
        self.cur = abs(h[-1] - h[0]) / (self.n * self.dt) * RPM if len(h) == self.n + 1 else 0.0
        self.gain = 0.0
        if self.last_row is not None and self.row > self.last_row:
            self.stopped = True  # past until_s: frozen (TS: `row > lastRow` returns)
        if not self.stopped:
            if self.cur > self.best:
                self.gain, self.best = self.cur - self.best, self.cur
            if out:
                self.stopped = True
        self.row += 1

    @property
    def value(self) -> float:
        return self.best

    @property
    def settled(self) -> bool:
        return self.stopped

    def columns(self) -> dict:
        return dict(rpm05=self.cur, peak=self.best, dpeak=self.gain, stopped=float(self.stopped))


class MaxDistance(NoMirror):
    """max_distance (the races: Sprint, Hurdles): the best, over full windows of n rows in which every row is upright, of
    the window minimum of the trunk's (x, y) - spawn projected on spawn.yaw (metrics.ts UprightWindowMax, n =
    HEIGHT_WINDOW_TICKS = 25 rows = 0.5 s); upright = tilt < 32 deg, tilt = acos(trunk z . world z) as episode.ts
    computes it (the env passes `cos_tilt`). Always settled. Also the race's finish (episode.ts's `success` event): the
    first row whose metric meets the Challenge's bar (metric_at_least, the finish line), its t = row * dt. A race Challenge has no
    floor_out_before_s, so the mirror's own value is the TS metric (no zeroing to apply)."""

    COLUMNS = ["dist", "finished", "finish_now", "finish_t"]
    DOC = {"dist": "the trunk's progress from the spawn along spawn.yaw this row (m; upright or not, no window)",
           "finished": "1 once the race is finished: the metric met the Challenge's bar (the TS `success` event)",
           "finish_now": "1 on the finishing row only",
           "finish_t": "the finish time (s, the sim's finishT) once finished, else the miss value (the Challenge's duration, "
                       "30 s: info ranking.miss_s, D-19); at the episode's end = the seed's race result"}
    WINDOW = 25  # packages/sim/src/runtime/constants.ts HEIGHT_WINDOW_TICKS
    UPRIGHT_DEG = 32.0  # constants.ts UPRIGHT_TILT_DEG
    MISS_S = 30.0  # the fallback; info ranking.miss_s (@duckbench/schema challengeRanking: the Challenge's duration_s) wins

    def __init__(self, info: dict, dt: float):
        super().__init__(info, dt)
        self.dt = dt
        self.n = int(info["metric"]["params"].get("window_rows", self.WINDOW))
        sp = info["spawn"]
        self.sx, self.sy = float(sp["x"]), float(sp["y"])
        self.ux, self.uy = math.cos(float(sp["yaw"])), math.sin(float(sp["yaw"]))
        self.MISS_S = float((info.get("ranking") or {}).get("miss_s") or info.get("duration_s") or self.MISS_S)
        s = info.get("finish") or info.get("success") or {}
        self.bar = float(s["metric_at_least"]) if "metric_at_least" in s else math.inf
        self.reset()

    def reset(self) -> None:
        self.row, self.buf, self.run, self.best, self.d = 0, deque(maxlen=self.n), 0, 0.0, 0.0
        self.finish_t, self.now = None, False

    def restore(self, tail: dict | None, row: int) -> None:
        self.reset()
        self.row = row
        if tail:
            for v in tail["hist"]:
                self.buf.append(float(v))
            self.run = int(tail["run"])
            self.best = float(tail["best"])
            self.finish_t = None if tail.get("finish_t") is None else float(tail["finish_t"])

    def push(self, qpos, qvel, cos_tilt: float = 1.0, **_kw) -> None:
        self.d = (float(qpos[0]) - self.sx) * self.ux + (float(qpos[1]) - self.sy) * self.uy
        tilt = math.acos(max(-1.0, min(1.0, cos_tilt))) * 180.0 / math.pi
        self.buf.append(self.d)
        self.run = self.run + 1 if tilt < self.UPRIGHT_DEG else 0
        if self.run >= self.n:
            self.best = max(self.best, min(self.buf))
        self.now = self.finish_t is None and self.best >= self.bar
        if self.now:
            self.finish_t = self.row * self.dt
        self.row += 1

    @property
    def value(self) -> float:
        return self.best

    @property
    def settled(self) -> bool:
        return True

    def columns(self) -> dict:
        return dict(dist=self.d, finished=float(self.finish_t is not None), finish_now=float(self.now),
                    finish_t=self.MISS_S if self.finish_t is None else self.finish_t)


TORSO_JSON = Path(__file__).resolve().with_name("torso_points.json")  # = packages/sim/src/runtime/torso-points.json


def torso_points() -> np.ndarray:
    """The torso's points [n, 3] in the trunk frame, m (the kit's copy of the sim's torso-points.json; in the monorepo
    it must equal the sim's, else this refuses: a stale copy would score a different torso)."""
    pts = json.loads(TORSO_JSON.read_text())
    try:
        from . import bridge
        repo = bridge.REPO
    except Exception:  # noqa: BLE001
        repo = None
    if repo is not None:
        src = Path(repo) / "packages" / "sim" / "src" / "runtime" / "torso-points.json"
        if src.exists() and json.loads(src.read_text()) != pts:
            raise SystemExit(f"{TORSO_JSON} differs from {src}: copy it over (cp {src} {TORSO_JSON})")
    return np.asarray(pts, dtype=np.float64).reshape(-1, 3)


def trunk_rotation(qpos) -> tuple:
    """The trunk's xmat (row-major 3x3) from its free-joint quaternion qpos[3:7] (w, x, y, z), as mj_kinematics makes
    it: mju_normalize4 (multiply by 1/norm when |norm - 1| > mjMINVAL; [1, 0, 0, 0] under mjMINVAL), then mju_quat2Mat
    (the identity quaternion gives the identity), in the same float order, so it is bit-identical to the sim's xmat."""
    w, x, y, z = float(qpos[3]), float(qpos[4]), float(qpos[5]), float(qpos[6])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-15:
        w, x, y, z = 1.0, 0.0, 0.0, 0.0
    elif abs(n - 1.0) > 1e-15:
        inv = 1.0 / n
        w, x, y, z = w * inv, x * inv, y * inv, z * inv
    if w == 1.0 and x == 0.0 and y == 0.0 and z == 0.0:
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    q00, q01, q02, q03 = w * w, w * x, w * y, w * z
    q11, q12, q13 = x * x, x * y, x * z
    q22, q23, q33 = y * y, y * z, z * z
    return (q00 + q11 - q22 - q33, 2 * (q12 - q03), 2 * (q13 + q02),
            2 * (q12 + q03), q00 - q11 + q22 - q33, 2 * (q23 - q01),
            2 * (q13 - q02), 2 * (q23 + q01), q00 - q11 - q22 + q33)


class TorsoReach(NoMirror):
    """torso_reach (the races since LB-168: Sprint v6, Hurdles v4, Ducky Steps v4; metrics.ts `torso_reach`): each row,
    reach = (trunk position - spawn) . u + max_i (R p_i) . u, u = (cos spawn.yaw, sin spawn.yaw, 0), R the trunk's
    rotation (its free-joint quaternion, trunk_rotation), p_i the torso's points (torso_points.json: the trunk body's
    meshes without the head or legs); the metric = max(0, the best reach so far), in any pose (no upright rule, no
    window). The race's finish (episode.ts's `success` event): the first row whose reach meets the line (finish
    metric_at_least), its time interpolated between the previous row and this one (prev.t + (line - prev.reach) /
    (reach - prev.reach) x (t - prev.t); this row's t with no previous row or no gain). Always settled. The columns are
    MaxDistance's, so race configs read the same names."""

    COLUMNS = ["dist", "finished", "finish_now", "finish_t"]
    DOC = {"dist": "the torso's reach this row: how far its furthest point is from the spawn along spawn.yaw (m; any pose)",
           "finished": "1 once the race is finished: the torso crossed the Challenge's line (the TS `success` event)",
           "finish_now": "1 on the finishing row only",
           "finish_t": "the finish time (s, interpolated between the two rows around the crossing, the sim's finishT) "
                       "once finished, else the miss value (the Challenge's duration, 30 s: info ranking.miss_s, D-19); "
                       "at the episode's end = the seed's race result"}
    MISS_S = 30.0
    _PTS: np.ndarray | None = None

    def __init__(self, info: dict, dt: float):
        super().__init__(info, dt)
        if TorsoReach._PTS is None:
            TorsoReach._PTS = torso_points()
        P = TorsoReach._PTS
        self.px, self.py, self.pz = P[:, 0].copy(), P[:, 1].copy(), P[:, 2].copy()
        self.tps = int(round(1.0 / dt))  # t = row / TICKS_PER_S, as episode.ts (not row * dt: a different rounding)
        sp = info["spawn"]
        self.sx, self.sy = float(sp["x"]), float(sp["y"])
        self.ux, self.uy = math.cos(float(sp["yaw"])), math.sin(float(sp["yaw"]))
        self.MISS_S = float((info.get("ranking") or {}).get("miss_s") or info.get("duration_s") or self.MISS_S)
        s = info.get("finish") or info.get("success") or {}
        self.line = float(s["metric_at_least"]) if "metric_at_least" in s else None
        self.reset()

    def reset(self) -> None:
        self.row, self.best, self.d = 0, -math.inf, 0.0
        self.prev_t, self.prev_reach = None, None
        self.finish_t, self.now = None, False

    def restore(self, tail: dict | None, row: int) -> None:
        self.reset()
        self.row = row
        if tail:
            self.best = -math.inf if tail.get("best") is None else float(tail["best"])
            if tail.get("prev_t") is not None:
                self.prev_t, self.prev_reach = float(tail["prev_t"]), float(tail["prev_reach"])
            self.finish_t = None if tail.get("finish_t") is None else float(tail["finish_t"])

    def reach_of(self, qpos) -> float:
        m = trunk_rotation(qpos)
        ux, uy = self.ux, self.uy
        ax = m[0] * ux + m[3] * uy  # R^T u: a point's reach is p . (R^T u)
        ay = m[1] * ux + m[4] * uy
        az = m[2] * ux + m[5] * uy
        front = float(np.max(self.px * ax + self.py * ay + self.pz * az))  # elementwise, no FMA: as the sim's loop
        return (float(qpos[0]) - self.sx) * ux + (float(qpos[1]) - self.sy) * uy + front

    def push(self, qpos, qvel, **_kw) -> None:
        t = self.row / self.tps
        reach = self.reach_of(qpos)
        self.d = reach
        self.now = False
        if self.finish_t is None and self.line is not None and reach >= self.line:
            if self.prev_t is None or reach <= self.prev_reach:
                self.finish_t = t
            else:
                self.finish_t = self.prev_t + ((self.line - self.prev_reach) / (reach - self.prev_reach)) * (t - self.prev_t)
            self.now = True
        self.best = max(self.best, reach)
        self.prev_t, self.prev_reach = t, reach
        self.row += 1

    @property
    def value(self) -> float:
        return max(0.0, self.best)

    @property
    def settled(self) -> bool:
        return True

    def columns(self) -> dict:
        return dict(dist=self.d, finished=float(self.finish_t is not None), finish_now=float(self.now),
                    finish_t=self.MISS_S if self.finish_t is None else self.finish_t)


# match.ts: the loser's lost_by for a win event (the rest keep the event's name)
LOST_BY = {"reached_zone": "rival_reached", "finished_first": "rival_finished", "goal_scored": "rival_scored",
           "ate_three": "rival_ate", "held_hill": "rival_held"}


class MatchPoints(NoMirror):
    """match_points (a duel Challenge, PvP: Sumo, and the races Maze race and Cliff): the judge of packages/sim/src/runtime/match.ts runMatch, one push per 50 Hz
    row k (row 0 = the reset state) with both duck slots' trunk positions (qpos at info duel.ducks[i].root_qpos_adr) and
    cos tilts (trunk xmat[8]). Rules from info `duel.rules`, constants from `duel.runtime`, nothing hard-coded:
      down      per slot, the raw fall state: set once tilt > fallen_tilt_deg for fallen_ticks rows in a row, cleared
                when upright (tilt < upright_tilt_deg) again; down_rows counts the rows since it was set;
      out       out "fell_off": the trunk centre below the platform top - fell_off_drop_m ("fell_off"), or, after the
                shrink's from_s, its xy distance from the ring centre beyond the ring radius ("red_zone"); out
                "trunk_outside_ring": beyond the ring radius ("out"); the radius lerps radius -> radius_end over the shrink;
      low       DuelRules.low_out (Wobbly Raft v3; absent = nothing computed): per slot, the rows in a row its trunk centre
                was under height_m above the stage top (runtime low_ref, sim lowOutRef / heightAboveStage: along the
                raft's normal, its ball-joint quaternion in qpos, from its pivot, less the pivot depth; quat None = the
                plane z = base); more than round(for_s * ticks_per_s) rows is an out ("too_low"), after the other outs;
      decision  until decided, per row: a slot out loses (both out on one row: "both_out", a draw; both by low_out:
                "both_low"), out decides before
                down; then down for more than down_loses_s (round(down_loses_s * ticks_per_s) rows) loses (both:
                "both_down"); at row max_ticks the time-out: the closer trunk (xy) to the ring centre wins, within
                draw_band_m (or timeout "draw") a draw; timeout "both_lose": no winner, reason "timeout_both_lose", 0
                points each. A decision on row k sets stop_row = min(max_ticks, k + the tail's rows): the episode runs
                on to stop_row, nothing after the decision changes it;
      win       DuelRules.win (match.ts judgeWin), decided after the outs and before down: None (Sumo) or "goose_chase"
                (slot 0 the goose, slot 1 the runner: the goose's trunk within chase.nogo_r of the pond centre loses,
                then a duck-duck touch (the caller's `touching` per row) wins for the goose, then the runner's trunk in
                the pond (chase.pond r) wins for it) or "bread" (Bread Fight, kit 1.10.0: the crumb field of
                duck_rl/crumbs.py, set per episode with reset(crumbs=...), is updated inside push on every row until the
                decision, after the posture rows (its eat test reads the raw `down`) and before the judge, with both
                slots' heads (xy) from the caller; the to_win-th crumb wins, "ate_three"; at row max_ticks unequal
                counts: the larger wins, "more_crumbs", else the time-out rule); other win rules are refused (ValueError);
      invalid   MuJoCo instability (the caller's `unstable()`: BADQACC/BADQPOS/BADQVEL warnings so far) checked when
                deciding voids the episode; a non-finite root qpos before the decision too (the caller calls
                invalidate(rows)): no winner, 0 points each, reason "invalid".
    value = slot 0's points once decided (1 win, 0.5 draw, 0 loss or invalid), else NaN; settled = decided. points,
    winner (slot or None), reason, t (the decision row's time), lost_by (per slot), stop_row; per-row state for rewards:
    dist / z (above the platform top) / upright / down / down_rows per slot, radius."""

    COLUMNS = ["ring", "dist0", "dist1", "down_rows0", "down_rows1", "decided", "points0", "points1"]
    DOC = {"ring": "the safe circle's radius this row (m; shrinks over the duel's shrink window)",
           "dist0, dist1": "each slot's trunk xy distance from the ring centre (m)",
           "down_rows0, down_rows1": "rows since the slot went down (the raw fall state; 0 while not down)",
           "decided": "1 once the judge decided (a win, a draw, or invalid)",
           "points0, points1": "each slot's points once decided (1 win, 0.5 draw, 0 loss/invalid), else 0"}

    def __init__(self, info: dict, dt: float):
        super().__init__(info, dt)
        duel = info["duel"]
        r, rt = duel["rules"], duel["runtime"]
        self.rules = r
        self.tps = int(rt["ticks_per_s"])
        ring = r["ring"]
        self.cx, self.cy, self.r0 = float(ring["x"]), float(ring["y"]), float(ring["radius"])
        self.shrink = ring.get("shrink")
        self.out_kind = r["out"]
        plat = r.get("platform")
        self.base = float(plat["height"]) if plat else 0.0
        fell = self.out_kind == "fell_off"
        self.fell_below = (self.base - float(rt["fell_off_drop_m"])) if fell else -math.inf
        self.red_from = (float(self.shrink["from_s"]) if self.shrink else math.inf) if fell else -math.inf
        self.limit_rows = None if r["down_loses_s"] is None else js_round(float(r["down_loses_s"]) * self.tps)
        self.tail_rows = js_round(float(r["tail_s"]) * self.tps)
        self.n_ticks = int(rt["max_ticks"])
        self.timeout_draw, self.draw_band = r["timeout"] == "draw", float(r["draw_band_m"])
        self.upright_deg, self.fallen_deg = float(rt["upright_tilt_deg"]), float(rt["fallen_tilt_deg"])
        self.fallen_ticks = int(rt["fallen_ticks"])
        self.qadr = [int(d["root_qpos_adr"]) for d in duel["ducks"]]
        fin = r.get("finish")  # a race (Maze race, Cliff): the first trunk into the zone wins
        self.finish = None if fin is None else (float(fin["x"]), float(fin["y"]), float(fin["radius"]),
                                                -math.inf if fin.get("z_min") is None else float(fin["z_min"]))
        self.win = r.get("win")
        if self.win not in (None, "goose_chase", "bread"):
            raise ValueError(f"duel win rule {self.win!r} has no Python judge (ported: goose_chase, bread)")
        self.chase = r.get("chase") if self.win == "goose_chase" else None
        self.needs_touch = self.win == "goose_chase"  # the caller passes the duck-duck touch per row
        self.both_lose = r["timeout"] == "both_lose"
        lo = r.get("low_out")
        self.low_h = float(lo["height_m"]) if lo else None
        self.low_rows = js_round(float(lo["for_s"]) * self.tps) if lo else 0
        self.low_ref = rt["low_ref"] if lo else None
        self.reset()

    def reset(self, crumbs=None) -> None:
        """A new episode; `crumbs` = its crumb field (duck_rl/crumbs.py CrumbField, win "bread"), else None."""
        self.crumbs = crumbs
        self.ate = -1  # the slot that ate a crumb on the last row pushed, else -1
        self.crumb_t = [[], []]  # each slot's eat times (match.ts crumbT)
        self.row = 0
        self.fall_count, self.down, self.down_rows = [0, 0], [False, False], [0, 0]
        self.upright, self.cos_tilt = [True, True], [1.0, 1.0]
        self.x, self.y, self.z, self.dist = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
        self.radius = self.r0
        self.decided = self.invalid = False
        self.winner, self.reason, self.t, self.stop_row = None, None, None, self.n_ticks
        self.lost_by = [None, None]
        self.low_count = [0, 0]

    def height_above_stage(self, qpos, i: int) -> float:
        """sim match.ts heightAboveStage: slot i's trunk centre above the stage top (the same sums, in the same order)."""
        ref = self.low_ref
        x, y, z = self.x[i], self.y[i], self.z[i]
        if ref["quat"] is None:
            return z - float(ref["base"])
        a = int(ref["quat"])
        w, qx, qy, qz = float(qpos[a]), float(qpos[a + 1]), float(qpos[a + 2]), float(qpos[a + 3])
        nx, ny, nz = 2 * (qx * qz + w * qy), 2 * (qy * qz - w * qx), 1 - 2 * (qx * qx + qy * qy)
        px, py, pz = (float(v) for v in ref["pivot"])
        return nx * (x - px) + ny * (y - py) + nz * (z - pz) - float(ref["depth"])

    def restore(self, tail: dict | None, row: int) -> None:
        raise NotImplementedError("match_points has no bank tail: a match starts at the spawn")

    def ring_radius(self, t: float) -> float:
        """match.ts ringRadiusAt."""
        s, r = self.shrink, self.r0
        if s is None or t <= s["from_s"]:
            return r
        if t >= s["to_s"]:
            return float(s["radius_end"])
        return r + ((s["radius_end"] - r) * (t - s["from_s"])) / (s["to_s"] - s["from_s"])

    def decide_timeout(self, d0: float, d1: float):
        """match.ts decideTimeout: None = a draw, else the closer slot."""
        if self.timeout_draw or abs(d0 - d1) <= self.draw_band:
            return None
        return 0 if d0 < d1 else 1

    def _decide(self, w, reason: str, t: float, k: int, unstable) -> None:
        self.winner, self.reason, self.t, self.decided = w, reason, t, True
        self.stop_row = min(self.n_ticks, k + self.tail_rows)
        if unstable is not None and unstable():
            self.invalidate(k + 1)

    def invalidate(self, rows: int) -> None:
        """The episode broke: `rows` valid rows (a non-finite state on row k: rows = k; instability found when deciding
        on row k: rows = k + 1). No winner, 0 points each, end t = max(0, rows - 1) / ticks_per_s."""
        self.invalid = self.decided = True
        self.winner, self.reason, self.t = None, "invalid", max(0, rows - 1) / self.tps
        self.lost_by = ["invalid", "invalid"]

    def _won(self, touching: bool) -> list:
        """match.ts judgeWin, per slot: the extra win event this row (None = none)."""
        if self.win == "bread":  # the to_win-th crumb (crumbs.py lets one duck eat per row)
            c = self.crumbs.counts if self.crumbs is not None else None
            return [("ate_three" if c is not None and c[i] >= self.crumbs.to_win else None) for i in (0, 1)]
        if self.win != "goose_chase":
            return [None, None]
        p, nogo = self.chase["pond"], float(self.chase["nogo_r"])
        px, py = float(p["x"]), float(p["y"])
        if js_hypot2(self.x[0] - px, self.y[0] - py) <= nogo:
            return [None, "goose_in_pond"]
        if touching:
            return ["caught", None]
        if js_hypot2(self.x[1] - px, self.y[1] - py) <= float(p["r"]):
            return [None, "reached_pond"]
        return [None, None]

    def push(self, qpos, qvel=None, cos_tilt=(1.0, 1.0), unstable=None, touching: bool = False, heads=None,
             **_kw) -> None:
        """Row `self.row`: both slots' posture bookkeeping (match.ts observeRow), the ring radius, then (until decided) the
        crumbs (win "bread": `heads` = each slot's head body xy on this row) and the judge. `unstable`: () -> bool,
        called only when a decision is made. `touching`: the ducks touch on this row (read by win "goose_chase" only)."""
        k = self.row
        t = k / self.tps
        for i in (0, 1):
            a = self.qadr[i]
            self.x[i], self.y[i], self.z[i] = float(qpos[a]), float(qpos[a + 1]), float(qpos[a + 2])
            c = float(cos_tilt[i])
            self.cos_tilt[i] = c
            tilt = math.acos(max(-1.0, min(1.0, c))) * 180 / math.pi
            self.upright[i] = tilt < self.upright_deg
            self.fall_count[i] = self.fall_count[i] + 1 if tilt > self.fallen_deg else 0
            if not self.down[i] and self.fall_count[i] >= self.fallen_ticks:
                self.down[i], self.down_rows[i] = True, 0
            elif self.down[i] and self.upright[i]:
                self.down[i] = False
            elif self.down[i]:
                self.down_rows[i] += 1
            self.dist[i] = js_hypot2(self.x[i] - self.cx, self.y[i] - self.cy)
        self.ate = -1
        if not self.decided and self.crumbs is not None:  # match.ts: place a due crumb, then the nearest duck in reach eats
            h = heads
            self.ate = self.crumbs.update(t, [(self.x[i], self.y[i], float(h[i][0]), float(h[i][1]), self.down[i])
                                              for i in (0, 1)])
            if self.ate >= 0:
                self.crumb_t[self.ate].append(t)
        radius = self.radius = self.ring_radius(t)
        if not self.decided:
            dist = self.dist
            if self.out_kind == "fell_off":
                out = [("fell_off" if self.z[i] < self.fell_below else "red_zone" if t > self.red_from and dist[i] > radius
                        else None) for i in (0, 1)]
            else:
                out = ["out" if dist[i] > radius else None for i in (0, 1)]
            if self.low_h is not None:
                for i in (0, 1):
                    self.low_count[i] = self.low_count[i] + 1 if self.height_above_stage(qpos, i) < self.low_h else 0
                    if out[i] is None and self.low_count[i] > self.low_rows:
                        out[i] = "too_low"
            lim = self.limit_rows
            down = [lim is not None and self.down[i] and self.down_rows[i] > lim for i in (0, 1)]
            won = self._won(touching) if self.win is not None else (None, None)
            fin = self.finish
            home = [fin is not None and js_hypot2(self.x[i] - fin[0], self.y[i] - fin[1]) <= fin[2] and self.z[i] >= fin[3]
                    for i in (0, 1)]
            if home[0] or home[1]:  # match.ts: the finish decides before out and down
                if home[0] and home[1]:
                    self._decide(None, "both_finish", t, k, unstable)
                else:
                    loser = 1 if home[0] else 0
                    self.lost_by[loser] = "finish"
                    self._decide(1 - loser, "finish", t, k, unstable)
            elif out[0] is not None or out[1] is not None:
                if out[0] is not None and out[1] is not None:
                    both = "both_low" if out[0] == "too_low" and out[1] == "too_low" else "both_out"
                    self._decide(None, both, t, k, unstable)  # out decides before down
                else:
                    loser = 0 if out[0] is not None else 1
                    self.lost_by[loser] = out[loser]
                    self._decide(1 - loser, out[loser], t, k, unstable)
            elif won[0] is not None or won[1] is not None:  # an extra win event: after the outs, before down
                if won[0] is not None and won[1] is not None:  # (not reachable with goose_chase; match.ts's name)
                    self._decide(None, "both_reached", t, k, unstable)
                else:
                    w = 0 if won[0] is not None else 1
                    self.lost_by[1 - w] = LOST_BY.get(won[w], won[w])
                    self._decide(w, won[w], t, k, unstable)
            elif down[0] or down[1]:
                if down[0] and down[1]:
                    self._decide(None, "both_down", t, k, unstable)
                else:
                    loser = 0 if down[0] else 1
                    self.lost_by[loser] = "down"
                    self._decide(1 - loser, "down", t, k, unstable)
            elif k == self.n_ticks and self.crumbs is not None and self.crumbs.counts[0] != self.crumbs.counts[1]:
                w = 0 if self.crumbs.counts[0] > self.crumbs.counts[1] else 1  # Bread Fight at the cap: more crumbs wins
                self.lost_by[1 - w] = "timeout"
                self._decide(w, "more_crumbs", t, k, unstable)
            elif k == self.n_ticks and self.both_lose:
                self.lost_by = ["timeout", "timeout"]
                self._decide(None, "timeout_both_lose", t, k, unstable)
            elif k == self.n_ticks:
                w = self.decide_timeout(dist[0], dist[1])
                if w is not None:
                    self.lost_by[1 - w] = "timeout"
                self._decide(w, "timeout", t, k, unstable)
        self.row += 1

    @property
    def points(self) -> tuple[float, float]:
        """Each slot's points (match.ts sideResult): invalid 0 each, a draw 0.5 each, else 1 / 0; (0, 0) until decided."""
        if not self.decided or self.invalid or self.reason == "timeout_both_lose":
            return 0.0, 0.0
        if self.winner is None:
            return 0.5, 0.5
        return (1.0, 0.0) if self.winner == 0 else (0.0, 1.0)

    @property
    def value(self) -> float:
        return self.points[0] if self.decided else float("nan")

    @property
    def settled(self) -> bool:
        return self.decided

    def columns(self) -> dict:
        p = self.points
        return dict(ring=self.radius, dist0=self.dist[0], dist1=self.dist[1], down_rows0=float(self.down_rows[0] if self.down[0] else 0),
                    down_rows1=float(self.down_rows[1] if self.down[1] else 0), decided=float(self.decided),
                    points0=p[0], points1=p[1])


MIRRORS = {"disc_speed": DiscSpeed, "wheel_speed": WheelSpeed, "wheel_turns": WheelTurns, "disc_turns": DiscTurns, "disc_peak_rpm": DiscPeakRpm,
           "max_distance": MaxDistance, "torso_reach": TorsoReach, "match_points": MatchPoints}


def make_metric(info: dict, dt: float):
    return MIRRORS.get(info["metric"]["kind"], NoMirror)(info, dt)


def meets_success(info: dict, value: float) -> bool:
    """The Challenge's bar (schema ChallengeSuccess: metric_at_least / metric_at_most)."""
    s = info.get("finish") or info.get("success") or {}
    if "metric_at_least" in s:
        return value >= s["metric_at_least"]
    if "metric_at_most" in s:
        return value <= s["metric_at_most"]
    return False
