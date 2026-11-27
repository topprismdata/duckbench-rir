"""The duckscript's 10-Hz program tick inside a solo Challenge's training env (kit 1.15.0): the trained node's `feed` and
`head` holds, evaluated every program tick exactly as the TS eval does (packages/sim runtime/episode.ts: sensors on the
forwarded row, then ProgramRunner.tick, then feedValues / headTargets into the skill layer), so the trained node sees
the command its feed gives it and the head joints its holds set.

The env pins the trained node (a solo training episode is that node's stretch of the duckscript), so transitions are
evaluated (memory, `any`, `set` run as in TS) but not followed. State starts at the env's reset: from the spawn that is
the program's own start (the trained node is the start node: exactly TS); from a handoff bank state, the memory and
variables of the nodes before it start empty (an approximation: the bank does not carry them). The sensors' noise
(target_* / goal_*) is drawn from the env's own stream in training (TS draws it from the episode's seeded stream).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import mujoco
import numpy as np

from . import v2 as V2
from .program import CompiledProgram, MatchRuntime, Sensors


class _NpPrng:
    """prng.Xoshiro128's `uniform` over a numpy Generator (training noise)."""

    def __init__(self, rng):
        self.rng = rng

    def uniform(self, lo: float, hi: float) -> float:
        return float(self.rng.uniform(lo, hi))


class SoloProgram:
    """The program state of one solo env: `node_id` is the trained node (feed / head are read from it)."""

    def __init__(self, info: dict, compiled: dict, model: mujoco.MjModel, duck, rng, node_id: str):
        self.rt = MatchRuntime(info)
        self.prog = CompiledProgram(compiled, "program")
        if not self.prog.v2:
            raise ValueError("SoloProgram runs a duck-program/2 program (a v1 node has no feed or head)")
        self.node = self.prog.nodes[node_id]
        self.state = V2.ProgramState(self.prog)
        if info.get("routes"):
            self.state.world = V2.RouteWorld(info["routes"])
        free = int(model.body_jntadr[duck.trunk])
        shim = SimpleNamespace(root_qpos_adr=int(model.jnt_qposadr[free]), trunk=duck.trunk, gyro_adr=duck.gyro_adr,
                               camera_site=model.site(self.rt.raw["camera_site"]).id,
                               ray_site=model.site(self.rt.raw["ray_site"]).id)
        self.root = shim.root_qpos_adr
        tgt = info.get("target")
        self.sensors = Sensors(model, self.rt, self.prog, _NpPrng(rng), shim,
                               target_body=-1 if tgt is None else tgt, goal=info.get("goal"), height_base=0.0)
        self.active = bool(self.node.feed or self.node.head)
        self.fall_count = 0

    def reset(self, k: int) -> None:
        """At the env's reset on row k: fresh state; the trained node entered now (its `set` at the first tick)."""
        self.state.reset()
        self.sensors.prev_x = self.sensors.prev_y = math.nan
        self.fall_count = 0
        self.n_enter = k // self.rt.program_every
        self.state.enter(self.node, None)

    def observe_row(self, cos_tilt: float) -> None:
        """episode.ts: the `fallen` counter per row (tilt > 60 deg rows in a row)."""
        tilt = math.acos(max(-1.0, min(1.0, cos_tilt))) * 180 / math.pi
        self.fall_count = self.fall_count + 1 if tilt > self.rt.fallen_tilt_deg else 0

    def tick(self, k: int, d: mujoco.MjData, done: bool):
        """Row k's program tick (k % program_every == 0): (feed values, head targets or None)."""
        n = k // self.rt.program_every
        fallen = self.fall_count >= self.rt.fallen_ticks
        s = self.sensors.compute(d, n, self.n_enter, fallen, done)
        w = self.state.world
        if w is not None:
            w.x, w.y, w.heading = float(d.qpos[self.root]), float(d.qpos[self.root + 1]), s.get("heading", 0.0)
        self.state.tick(self.node, s)  # memory, any rules, set: as TS (the env stays on the trained node)
        feed = self.state.feed_values(self.node, s) if self.node.feed else []
        head = self.state.head_targets(self.node, s) if self.node.head else None
        if head is not None and not any(math.isfinite(v) for v in head):
            head = None
        return feed, head


def hold_head(action: np.ndarray, prev: np.ndarray, head, ema: float) -> None:
    """sim.ts DuckLoop.holdHead on a float32 action: actions 5-8 approached from the previous action at `ema`."""
    if head is None:
        return
    for i in range(4):
        v = head[i]
        if math.isfinite(v):
            p = float(prev[5 + i])
            action[5 + i] = p + ema * (v - p)
