"""Runtime mechanics of a solo Challenge in Python (kit 1.9.0): props the runtime moves itself and what the duckscript senses
of them, mirrored from packages/sim/src/runtime/*.ts so ChallengeEnv (duck_rl/env.py) runs the Challenge as runEpisode does.

  drives     prop-drive.ts bindDrives: JointDrive {joint, at(t), vel(t)?}; apply(d, t) writes the joint's qpos = at(t),
             zeroes its qvel and writes qvel = vel(t) when given. episode.ts applies at t = 0 inside the reset and at
             (k + 1) / TICKS_PER_S right before tick k (so row k + 1 holds the pose of its own time);
  mover      one per mechanic of the Challenge's info (ts/prepare.ts soloMechanics), created per episode from the Challenge's
             "<challenge>:<version>:<seed>:match" stream (episode.ts lightPrng; a training label otherwise): its drives, the
             signals it adds for the duckscript (program.py SoloSensors, at the 10-Hz program tick, rounded as
             sensors.ts rounds them: +x.toFixed(6)) and an optional per-row out check (a car hit, moving on red);
  Mechanics  the Challenge's movers in episode.ts's draw order on one shared stream (`light`, then `traffic`; a Challenge naming an
             unported mechanic is refused); make_mechanics(info, model) is None for a Challenge without one, and the env
             then runs exactly its old code path.

Traffic (runtime/traffic.ts, Crossy Duck): per lane, in order, from the stream: the speed, a weight per car for the gaps,
the phase; car j of lane i is at y = mod(phase + off_j + dir v t, span) - span / 2 and drives its slide `<id>_slide` with
its real speed; car_eta / car_eta2 / gap_eta (s, capped at 10) and kerb_dist (m, capped at 3) for the trunk's (x, y);
a contact with a car geom (a geom of a car prop's body) on a row before the finish = a hit (episode.ts carContact).
Light (runtime/light.ts + episode.ts, Red Light, Green Light; kit 1.9.0): the switch times drawn first from the stream
(drawLightSchedule: green_s, red_s alternating from a green start, each time +t.toFixed(6), up to the Challenge's duration),
the doll's hinge `<doll>_hinge` driven to dollAngle (smoothstep over max(grace_s, 0.2) s after each switch, qvel 0),
the signals light (1 green, 0 red) and light_t (s since the last switch, rounded), and the out rule: on red, from
grace_s after the switch, the trunk's (x, y) is pinned on the first such row of the phase and any later row of it more
than drift_m (Math.hypot) away is out ("moved on red"), before the finish only.
Float formulas, operation order, JS `%` (math.fmod) and Math.min / Math.max are kept as in TS; stdlib + numpy only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable

import mujoco
import numpy as np

from .jsmath import js_hypot2
from .prng import Xoshiro128, match_prng

TICKS_PER_S = 50  # runtime/constants.ts TICKS_PER_S (info runtime.ticks_per_s; checked in Mechanics)
_Q6 = Decimal("0.000001")


def fixed6(x: float) -> float:
    """JS `+x.toFixed(6)` for a finite |x| < 1e21: the decimal nearest x's exact value at 6 places, a tie away from zero
    (toFixed picks the larger n of |x|), parsed back to the nearest double (Python's round() ties to even instead)."""
    if x != x or x in (math.inf, -math.inf):
        return x
    return float(Decimal(x).quantize(_Q6, rounding=ROUND_HALF_UP))


def js_mod(a: float, m: float) -> float:
    """traffic.ts mod: ((a % m) + m) % m with JS `%` (math.fmod: the sign of the dividend)."""
    return math.fmod(math.fmod(a, m) + m, m)


# ---- prop-drive.ts ---------------------------------------------------------------------------------------------------
@dataclass
class JointDrive:
    joint: str
    at: Callable[[float], float]
    vel: Callable[[float], float] | None = None


class BoundDrives:
    """prop-drive.ts bindDrives for hinge and slide joints (1 qpos, 1 dof; ball and free drives are not ported)."""

    def __init__(self, model: mujoco.MjModel, drives: list[JointDrive]):
        self.bound = []
        for dr in drives:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, dr.joint)
            if j < 0:
                raise ValueError(f"prop drive: no joint {dr.joint!r} in the world")
            if int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                raise ValueError(f"prop drive {dr.joint!r}: only hinge and slide joints are ported")
            self.bound.append((dr.at, dr.vel, int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])))

    def apply(self, d: mujoco.MjData, t: float) -> None:
        qpos, qvel = d.qpos, d.qvel
        for at, vel, q, v in self.bound:
            qpos[q] = at(t)
            qvel[v] = 0.0
            if vel is not None:
                qvel[v] = vel(t)


# ---- traffic.ts ------------------------------------------------------------------------------------------------------
class Traffic:
    """runtime/traffic.ts createTraffic + episode.ts carGeoms / carContact (module doc). reset(prng) draws the seed's
    lanes; then drives, sense(t, x, y), signals(t, x, y, out), out(d)."""

    KEY = "traffic"
    SIGNALS = ("car_eta", "car_eta2", "gap_eta", "kerb_dist")
    COLUMNS = ["hit"]
    DOC = {"hit": "1 once a car touched the duck (Crossy Duck: out, the race result is the cap; the metric is 0 from "
                  "then on and `lost` is 1)"}
    DUCK_HW = 0.1  # traffic.ts TRAFFIC_DUCK_HW
    CAP_S = 10.0
    CAP_M = 3.0
    WHY = "hit"

    def __init__(self, info: dict, model: mujoco.MjModel):
        r = info["traffic"]
        self.span, self.L, self.min_gap_s = float(r["span"]), float(r["car_len"]), float(r["min_gap_s"])
        self.lanes = r["lanes"]
        self.park = {k: float(v) for k, v in r["park"].items()}
        self.reach = self.L / 2 + self.DUCK_HW
        self.runs: list[tuple[float, int, list[float], float]] = []
        bodies = []
        for c in (c for ln in self.lanes for c in ln["cars"]):
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, c)
            if b < 0:
                raise ValueError(f"traffic: no car body {c!r} in the world")
            bodies.append(b)
        self.car_geom = np.isin(model.geom_bodyid, bodies)
        dt = 1 / TICKS_PER_S
        self.drives = []
        for i, ln in enumerate(self.lanes):
            for j, cid in enumerate(ln["cars"]):
                park = self.park.get(cid, 0.0)
                self.drives.append(JointDrive(
                    f"{cid}_slide",
                    (lambda t, i=i, j=j, park=park: self.car_y(i, j, max(0.0, t - dt)) - park),
                    (lambda t, i=i: self.runs[i][1] * self.runs[i][0])))

    def reset(self, prng: Xoshiro128) -> None:
        span, L = self.span, self.L
        runs = []
        for ln in self.lanes:
            v = prng.uniform(float(ln["speed_min"]), float(ln["speed_max"]))
            k = len(ln["cars"])
            min_gap = self.min_gap_s * v
            free = max(0.0, span - k * L - k * min_gap)
            w = [prng.uniform(0.1, 1) for _ in ln["cars"]]
            sw = 0.0
            for x in w:  # (an explicit loop: Python 3.12's sum() of floats is compensated, JS reduce is not)
                sw = sw + x
            offs, s = [], 0.0
            for j in range(k):
                offs.append(s)
                s += L + min_gap + (free * w[j]) / sw
            runs.append((v, int(ln["dir"]), offs, prng.uniform(0, span)))
        self.runs = runs

    def car_y(self, i: int, j: int, t: float) -> float:
        v, dr, offs, phase = self.runs[i]
        return js_mod(phase + offs[j] + dr * v * t, self.span) - self.span / 2

    def lane_at(self, i: int, t: float, y: float) -> tuple[float, float]:
        v, dr, offs, _ = self.runs[i]
        span, reach = self.span, self.reach
        eta, clear = math.inf, 0.0
        for j in range(len(offs)):
            d = js_mod(dr * (y - self.car_y(i, j, t)), span)
            if d > span - reach:
                eta = 0.0
                clear = max(clear, (reach - (span - d)) / v)
            elif d <= reach:
                eta = 0.0
                clear = max(clear, (d + reach) / v)
            else:
                e = (d - reach) / v
                if e < eta:
                    eta = e
        if eta > 0:
            clear = eta + (2 * reach) / v
        return eta, clear

    def sense(self, t: float, x: float, y: float) -> tuple[float, float, float, float]:
        lanes = self.lanes
        a = next((i for i, ln in enumerate(lanes) if x <= ln["x"] + ln["hw"]), -1)
        if a < 0:
            return self.CAP_S, self.CAP_S, self.CAP_S, self.CAP_M
        la = lanes[a]
        kerb = max(0.0, la["x"] - la["hw"] - x)
        eta, clear = self.lane_at(a, t, y)
        nb = lanes[a + 1] if a + 1 < len(lanes) else None
        eta2 = self.lane_at(a + 1, t, y)[0] if nb is not None and nb["block"] == la["block"] else self.CAP_S
        gap = 0.0 if eta > 0 else clear
        return min(self.CAP_S, eta), min(self.CAP_S, eta2), min(self.CAP_S, gap), min(self.CAP_M, kerb)

    def signals(self, t: float, x: float, y: float, out: dict) -> None:
        """sensors.ts: the four values rounded to 6 places (all four, as TS computes them when any is read)."""
        e1, e2, g, k = self.sense(t, x, y)
        out["car_eta"], out["car_eta2"], out["gap_eta"], out["kerb_dist"] = fixed6(e1), fixed6(e2), fixed6(g), fixed6(k)

    def out(self, d: mujoco.MjData, t: float) -> bool:
        """episode.ts carContact: any contact this row with a car geom (cars touch nothing but the duck)."""
        n = d.ncon
        if n == 0:
            return False
        g = d.contact.geom[:n]
        return bool(np.any(self.car_geom[g[:, 0]] | self.car_geom[g[:, 1]]))


# ---- light.ts --------------------------------------------------------------------------------------------------------
class Light:
    """runtime/light.ts drawLightSchedule / lightAt / dollAngle + episode.ts's red-light pin (module doc). reset(prng)
    draws the seed's switch times; then drives, light_at(t), signals(t, x, y, out), out(d, t)."""

    KEY = "light"
    SIGNALS = ("light", "light_t")
    COLUMNS = ["red_out"]
    DOC = {"red_out": "1 once the duck moved on red (Red Light, Green Light: out, the race result is the cap; the metric "
                      "is 0 from then on and `lost` is 1)"}
    WHY = "moved_on_red"  # episode.ts's out event `why`

    def __init__(self, info: dict, model: mujoco.MjModel):
        r = info["light"]
        self.green_s = (float(r["green_s"][0]), float(r["green_s"][1]))
        self.red_s = (float(r["red_s"][0]), float(r["red_s"][1]))
        self.grace_s, self.drift_m = float(r["grace_s"]), float(r["drift_m"])
        self.duration_s = float(info["duration_s"])
        self.turn_s = max(self.grace_s, 0.2)
        self.switches: list[float] = []
        self.pin = [-1, 0.0, 0.0]
        self.drives = [] if r.get("doll") is None else [JointDrive(f"{r['doll']}_hinge", self.doll_angle)]

    def reset(self, prng: Xoshiro128) -> None:
        out, t, red = [], 0.0, False
        while t < self.duration_s:
            lo, hi = self.red_s if red else self.green_s
            t += prng.uniform(lo, hi)
            out.append(fixed6(t))
            red = not red
        self.switches = out
        self.pin = [-1, 0.0, 0.0]

    def light_at(self, t: float) -> tuple[bool, float, int]:
        """lightAt: (green, since, phase)."""
        sw = self.switches
        i = 0
        while i < len(sw) and t >= sw[i]:
            i += 1
        return i % 2 == 0, (0.0 if i == 0 else sw[i - 1]), i

    def doll_angle(self, t: float) -> float:
        green, since, phase = self.light_at(t)
        if phase == 0:
            return 0.0
        u = min(1.0, (t - since) / self.turn_s)
        e = u * u * (3 - 2 * u)
        return math.pi * (1 - e) if green else math.pi * e

    def signals(self, t: float, x: float, y: float, out: dict) -> None:
        """sensors.ts: light = 1 / 0, light_t = +(t - since).toFixed(6) (both, as TS computes them when either is read)."""
        green, since, _ = self.light_at(t)
        out["light"], out["light_t"] = (1 if green else 0), fixed6(t - since)

    def out(self, d: mujoco.MjData, t: float) -> bool:
        """episode.ts: on red from grace_s after the switch, pin the trunk on the phase's first row, out on a drift."""
        green, since, phase = self.light_at(t)
        if green or t < since + self.grace_s - 1e-9:
            return False
        x, y = float(d.qpos[0]), float(d.qpos[1])
        pin = self.pin
        if pin[0] != phase:
            pin[0], pin[1], pin[2] = phase, x, y
            return False
        return js_hypot2(x - pin[1], y - pin[2]) > self.drift_m


# episode.ts draws the mechanics from the one ":match" stream in this order (light, tiles, cups, sweeper, crumbs, wall,
# cannons, maze, traffic, wave, gust, ice, roomba); the kit ports these, in that order:
MOVERS = (Light, Traffic)
UNPORTED = ("tiles", "cups", "bread", "maze", "wave", "ice", "tide")  # Challenge keys a solo Challenge may carry


def mover_columns(info: dict) -> tuple[list[str], dict]:
    """The state columns (and their docs) of the Challenge's movers; ([], {}) for a Challenge without one (the old layout)."""
    cols, doc = [], {}
    for cls in MOVERS:
        if cls.KEY in info:
            cols += cls.COLUMNS
            doc.update(cls.DOC)
    return cols, doc


def mover_signals(info: dict) -> tuple[str, ...]:
    """The duckscript signals the Challenge's movers add (program.py accepts them in a solo program)."""
    return tuple(s for cls in MOVERS if cls.KEY in info for s in cls.SIGNALS)


class Mechanics:
    """The Challenge's movers (module doc). reset(label) = the episode's draws from match_prng(label, "match") (label: the
    episode label "<challenge>:<version>:<seed>" of a public seed, or a training label); apply(d, t) = every drive; signals(t,
    x, y, out) = every mover's senses; out(d, t) -> the first mover's why ("hit", "moved_on_red": the TS out event's) on a row one fires, else None
    (every mover's out rule runs on every row it is asked: the light's pin is per row state)."""

    def __init__(self, info: dict, model: mujoco.MjModel, movers: list):
        self.info, self.movers = info, movers
        rt = info.get("runtime") or {}
        if int(rt.get("ticks_per_s", TICKS_PER_S)) != TICKS_PER_S:
            raise ValueError(f"runtime ticks_per_s {rt.get('ticks_per_s')} != movers.TICKS_PER_S {TICKS_PER_S}")
        self.drives = BoundDrives(model, [dr for mv in movers for dr in mv.drives])
        self.signal_names = tuple(s for mv in movers for s in mv.SIGNALS)
        self.label = None

    def reset(self, label: str) -> None:
        self.label = label
        prng = match_prng(label, "match")
        for mv in self.movers:
            mv.reset(prng)

    def apply(self, d: mujoco.MjData, t: float) -> None:
        self.drives.apply(d, t)

    def signals(self, t: float, x: float, y: float, out: dict) -> None:
        for mv in self.movers:
            mv.signals(t, x, y, out)

    def out(self, d: mujoco.MjData, t: float) -> str | None:
        why = None
        for mv in self.movers:
            if mv.out(d, t) and why is None:
                why = mv.WHY
        return why


def make_mechanics(info: dict, model: mujoco.MjModel) -> Mechanics | None:
    """The Challenge's Mechanics, or None for a Challenge without a ported mechanic (the env's old path). ValueError for a Challenge
    that also names an unported one (their shared stream's draws would be off)."""
    movers = [cls(info, model) for cls in MOVERS if cls.KEY in info]
    if not movers:
        return None
    bad = [k for k in UNPORTED if k in info]
    if bad:
        raise ValueError(f"Challenge {info['challenge']['id']}: mechanics {bad} are not ported to the kit (duck_rl/movers.py)")
    return Mechanics(info, model, movers)
