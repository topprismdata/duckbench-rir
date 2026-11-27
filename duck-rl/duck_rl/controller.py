"""Command controllers: the TS skills layer (packages/sim/src/runtime/skills.ts + constants.ts WALK) in Python.

WalkController is the walk-stand host (a walk node through the walk controller, a stand node zero twist);
RawController is the raw host (skills.ts #raw: the move's own 13-value block, verbatim); command_vector mirrors the
schema's commandVectorOf (packages/schema/src/strategy.ts) for named blocks.

A controller produces the 13-D command the policy is fed each policy tick (before the head EMA, which the env's obs
builder applies, as sim.ts DuckLoop.observe does). Call order per tick, as runtime/episode.ts does:
    tick 0:  reset()            (layer.enter at k = 0: no observe)
    tick j>0: observe(gyro_z)   (layer.observe on the state after tick j-1: gyro sensor z of s_j)
    then:    command()          (layer.step -> #command)
"""
from __future__ import annotations

import math

import numpy as np

POLICY_DT = 0.02
# constants.ts WALK
WALK_FORWARD = [(0.1, 0.039), (0.15, 0.065), (0.2, 0.092), (0.25, 0.118), (0.3, 0.144), (0.35, 0.166), (0.4, 0.19)]
WALK_BACKWARD = [(-0.3, -0.161), (-0.2, -0.07)]
LATERAL_GAIN, LATERAL_MAX = 5.0, 0.3
KICK_COMMAND, KICK_TICKS = 0.4, 25
YAW_GAIN, WZ_LIMIT = 2.0, 1.0
# packages/schema/src/strategy.ts COMMAND_SLOTS: the 13 command slots in obs order (obs[48:61]), raw policy units
COMMAND_SLOTS = ("vx", "vy", "wz", "neck_pitch", "head_pitch", "head_yaw", "head_roll",
                 "body_0", "body_1", "body_2", "body_3", "body_4", "body_5")


def command_vector(value, knobs=()) -> np.ndarray:
    """A raw command block -> 13 floats (the schema's commandVectorOf): 13 numbers as they are, or {name: value} where
    a name is a slot name or a knob's name (knobs: [{name, slot}], e.g. a card's host.knobs); unnamed slots are 0.
    ValueError on a wrong length, a non-finite value, an unknown name, or two names for one slot."""
    if isinstance(value, (list, tuple, np.ndarray)):
        v = np.asarray(value, float)
        if v.shape != (13,) or not np.all(np.isfinite(v)):
            raise ValueError(f"a command block is 13 finite numbers ({', '.join(COMMAND_SLOTS)}), got {value!r}")
        return v.copy()
    if not isinstance(value, dict):
        raise ValueError(f"a command block is 13 numbers or {{name: number}}, got {value!r}")
    slot_of = {s: i for i, s in enumerate(COMMAND_SLOTS)}
    knob_slot = {}
    for k in knobs:
        if k["slot"] not in slot_of:
            raise ValueError(f"knob {k['name']!r}: no command slot {k['slot']!r}")
        if k["name"] in knob_slot:
            raise ValueError(f"knob {k['name']!r} is declared twice")
        if k["name"] in slot_of and k["name"] != k["slot"]:
            raise ValueError(f"knob {k['name']!r} is named after the slot {k['name']!r} but maps onto {k['slot']!r}")
        knob_slot[k["name"]] = slot_of[k["slot"]]
    out, by = np.zeros(13), {}
    for name, x in value.items():
        i = knob_slot.get(name, slot_of.get(name))
        if i is None:
            raise ValueError(f"unknown command name {name!r} (the names: {', '.join([*knob_slot, *(s for s in COMMAND_SLOTS if s not in knob_slot)])})")
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x):
            raise ValueError(f"command {name!r} is not a finite number")
        if i in by:
            raise ValueError(f"{name!r} and {by[i]!r} both set the slot {COMMAND_SLOTS[i]}")
        by[i] = name
        out[i] = float(x)
    return out


def _interp(points, x: float) -> float:
    """skills.ts interp: linear over points sorted by x, clamped to the ends."""
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for i in range(1, len(points)):
        x1, y1 = points[i]
        if x <= x1:
            x0, y0 = points[i - 1]
            return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0)
    return points[-1][1]


_FWD = [(b, a) for a, b in WALK_FORWARD]
_BACK = [(b, a) for a, b in WALK_BACKWARD]


def walk_command_vx(vx: float) -> float:
    """Real forward speed (m/s) -> alpha_walking vx command (skills.ts walkCommandVx). 0.19 m/s -> 0.4."""
    if vx == 0:
        return 0.0
    return _interp(_FWD if vx > 0 else _BACK, vx)


def _clip(x: float, lim: float) -> float:
    return max(-lim, min(lim, x))


class WalkController:
    """One `walk` node entered at tick 0 from a stop (the TS program `walk.json` / `walk-noget.json` start node):
    0.5 s start kick at vx 0.4, then walkCommandVx(vx); vy * 5 clipped to 0.3; the gyro heading hold
    wz_cmd = clip(wz + 2 (wz k dt - psi_hat), +-1) with psi_hat += gyro_z dt per observed tick; head targets
    [neck 0, head_pitch, head_yaw, head_roll]; body 0. Policy: alpha_walking (twist + head + body in its obs)."""

    policy = "alpha_walking"

    def __init__(self, vx: float = 0.19, vy: float = 0.0, wz: float = 0.0, head_pitch: float = 0.0,
                 head_yaw: float = 0.0, head_roll: float = 0.0, walk: bool = True):
        self.vx, self.vy, self.wz = float(vx), float(vy), float(wz)
        self.head = (0.0, float(head_pitch), float(head_yaw), float(head_roll))
        self.moving = self.vx != 0 or self.vy != 0 or self.wz != 0
        self.walk = walk  # a stand node (no feed of its speeds)
        self.reset()

    def reset(self) -> None:
        self.k, self.psi = 0, 0.0
        self.kick = KICK_TICKS if self.moving else 0  # entered from a stop (#moving false at the episode start)
        self.fed = None  # duck-program/2 feed on a stock walk (kit 1.15.0): [vx, vy, wz], clamped (skills.ts feed)
        self.yaw_ref = 0.0
        self.was_moving = False  # skills.ts #moving: the last tick's command had a twist

    def feed(self, values, ranges) -> None:
        """skills.ts feed on a stock walk: the fed vx vy wz (slots 0-2, clamped to the walk's ranges) replace the params
        from now on; other slots are ignored (a raw move takes them)."""
        if not self.walk or not values:
            return
        f = self.fed if self.fed is not None else [self.vx, self.vy, self.wz]
        for i, v in values:
            if i <= 2:
                lo, hi = ranges[i]
                f[i] = 0.0 if not math.isfinite(v) else min(max(v, lo), hi)
        self.fed = f

    def observe(self, gyro_z: float) -> None:
        self.k += 1
        self.psi += gyro_z * POLICY_DT

    def command(self) -> np.ndarray:
        c = np.zeros(13)
        c[3:7] = self.head
        fed = self.fed
        if fed is None:
            if self.moving:
                k = self.k
                c[0] = (KICK_COMMAND if self.vx >= 0 else -KICK_COMMAND) if k < self.kick else walk_command_vx(self.vx)
                c[1] = _clip(self.vy * LATERAL_GAIN, LATERAL_MAX)
                c[2] = _clip(self.wz + YAW_GAIN * (self.wz * k * POLICY_DT - self.psi), WZ_LIMIT)
            self.was_moving = self.moving
            return c
        vx, vy, wz = fed  # skills.ts #command with a feed: the integrated yaw reference, the start kick on a fed start
        self.yaw_ref += wz * POLICY_DT
        moving = vx != 0 or vy != 0 or wz != 0
        if moving:
            k = self.k
            if not self.was_moving and vx != 0 and k >= self.kick:
                self.kick = k + KICK_TICKS
            c[0] = (KICK_COMMAND if vx >= 0 else -KICK_COMMAND) if k < self.kick else walk_command_vx(vx)
            c[1] = _clip(vy * LATERAL_GAIN, LATERAL_MAX)
            c[2] = _clip(wz + YAW_GAIN * (self.yaw_ref - self.psi), WZ_LIMIT)
        self.was_moving = moving
        return c

    def spec(self) -> dict:
        return dict(kind="walk", vx=self.vx, vy=self.vy, wz=self.wz, head_pitch=self.head[1], head_yaw=self.head[2],
                    head_roll=self.head[3])


class RawController:
    """The raw host (skills.ts #raw): the move's own 13-value block, verbatim, on every tick: no speed map, start kick or
    yaw loop, head and body in the obs (ChallengeEnv turns both obs masks on). The head slots still pass the obs builder's
    head EMA (nb_obs: h += 0.2 (target - h) per tick, its state carried across nodes), as TS DuckLoop.observe does for
    every policy: a head value is a target the move sees approached. `kick` exists only so the env's walk -> walk
    switch code can clear it."""

    def __init__(self, command):
        self.cmd = command_vector(command)
        self.base = self.cmd.copy()
        self.kick = 0

    def reset(self) -> None:
        self.cmd = self.base.copy()

    def feed(self, values, ranges=None) -> None:
        """skills.ts feed on a raw move: the move's own block with the fed slots replaced (until the next feed)."""
        if not values:
            return
        c = self.base.copy()
        for i, v in values:
            c[i] = v
        self.cmd = c

    def observe(self, gyro_z: float) -> None:
        pass

    def command(self) -> np.ndarray:
        return self.cmd.copy()

    def spec(self) -> dict:
        return dict(kind="raw", command=self.cmd.tolist())


class ZeroController:
    """All-zero command block, head and body targets: what TS feeds a roulade (and a stand node without head params,
    programs/stand-forever.json, the swap harness for a continuously running policy)."""

    def reset(self) -> None:
        pass

    def observe(self, gyro_z: float) -> None:
        pass

    def command(self) -> np.ndarray:
        return np.zeros(13)

    def spec(self) -> dict:
        return dict(kind="zero")


def make_controller(spec: dict):
    spec = dict(spec)
    kind = spec.pop("kind")
    if kind == "walk":
        return WalkController(**spec)
    if kind == "zero":
        return ZeroController()
    if kind == "raw":
        return RawController(spec["command"])
    raise ValueError(f"unknown controller {kind!r}")
