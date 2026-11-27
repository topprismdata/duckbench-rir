"""Per-Challenge adapters of the training env (duck_rl/env.py ChallengeEnv), keyed by Challenge id, with a generic fallback.

An adapter owns what is specific to a Challenge's props: training-only prop randomisation, an initial prop spin, the default
episode end, and extra state columns (with aliases and docs). The metric is not here: it is per metric kind
(duck_rl/metrics.py). A Challenge with no adapter gets GenericAdapter: with a hinged prop (info.props) the first one's
generic columns and randomisation (HingedAdapter), else nothing; the episode then ends at the Challenge's duration.

Adapter API (all optional to override):
  COLUMNS   extra state columns this adapter fills; ALIASES {alias: column}; DOC {name: meaning}
  SCALE_KEY / RPM0_KEY   the cfg keys of the prop randomisation / initial spin (the generic `prop_scale` / `prop_rpm0`
            are accepted too)
  default_end_s()        the episode end when cfg has no end_s
  reset(train, spawn)    before the episode's draws are applied (prop params: mj_setConst then sees them)
  spin(train)            after a spawn reset (an initial prop speed)
  fill(st, EX, d)        after each step: write COLUMNS into the state row st (EX = column index by name)
A duel Challenge's adapter (sumo.py) serves duck_rl/duel_env.py DuelEnv instead: COLUMNS, DOC, default_end_s, and
  fill(st, EX, row)      after each step, from the world's DuelRow (the learner's side first)
  priv(row, out)         the critic's 6 extras of a row (the reset row too)
To add a Challenge: a module here with an Adapter subclass, registered in ADAPTERS.
"""
from __future__ import annotations

import math

import numpy as np

RPM = 60.0 / (2.0 * math.pi)


class GenericAdapter:
    COLUMNS: list[str] = []
    ALIASES: dict[str, str] = {}
    DOC: dict[str, str] = {}

    def __init__(self, env):
        self.env, self.cfg, self.info = env, env.cfg, env.info

    def default_end_s(self) -> float:
        return float(self.info["duration_s"])

    def reset(self, train: bool, spawn: bool) -> None:
        pass

    def spin(self, train: bool) -> None:
        pass

    def fill(self, st, EX, d) -> None:
        pass


class HingedAdapter(GenericAdapter):
    """The first hinged prop (info.props[0]): its speed/angle columns (prop_*), training-only bearing/mass scales and an
    initial spin on spawn starts."""

    KIND: str | None = None  # the prop kind this adapter is for (None: the first hinged prop)
    SCALE_KEY, RPM0_KEY = "prop_scale", "prop_rpm0"
    COLUMNS = ["prop_omega", "prop_rpm", "prop_angle", "prop_turns"]
    DOC = {"prop_omega": "the hinged prop's speed (rad/s, signed)", "prop_rpm": "the same in rpm (signed)",
           "prop_angle": "its hinge angle (rad, unwrapped)", "prop_turns": "its hinge angle in turns"}

    def __init__(self, env):
        super().__init__(env)
        props = [p for p in self.info["props"] if self.KIND is None or p["kind"] == self.KIND]
        self.prop = p = props[0]
        m = env.duck.model
        self.q, self.v = int(p["qpos_adr"]), int(p["dof_adr"])
        j = int(np.nonzero(m.jnt_qposadr == self.q)[0][0])
        import mujoco
        assert mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) == p["hinge"], f"{p['hinge']} address mismatch"
        self.body = int(m.jnt_bodyid[j])
        self._p0 = dict(frictionloss=float(m.dof_frictionloss[self.v]), damping=float(m.dof_damping[self.v]),
                        mass=float(m.body_mass[self.body]), inertia=m.body_inertia[self.body].copy())

    def _cfg(self, key: str, generic: str):
        return self.cfg.get(key) if self.cfg.get(key) is not None else self.cfg.get(generic)

    def reset(self, train: bool, spawn: bool) -> None:
        m, p0, rng = self.env.duck.model, self._p0, self.env.rng
        sc = dict(frictionloss=1.0, damping=1.0, mass=1.0)
        if train:
            for key, (lo, hi) in (self._cfg(self.SCALE_KEY, "prop_scale") or {}).items():
                sc[key] = rng.uniform(lo, hi)
        m.dof_frictionloss[self.v] = p0["frictionloss"] * sc["frictionloss"]
        m.dof_damping[self.v] = p0["damping"] * sc["damping"]
        m.body_mass[self.body] = p0["mass"] * sc["mass"]
        m.body_inertia[self.body] = p0["inertia"] * sc["mass"]

    def spin(self, train: bool) -> None:
        if train:
            lo, hi = self._cfg(self.RPM0_KEY, "prop_rpm0") or (0.0, 0.0)
            self.env.duck.data.qvel[self.v] = (self.env.rng.uniform(lo, hi) if hi > lo else lo) / RPM

    def fill(self, st, EX, d) -> None:
        w = float(d.qvel[self.v])
        st[EX["prop_omega"]], st[EX["prop_rpm"]] = w, w * RPM
        st[EX["prop_angle"]], st[EX["prop_turns"]] = d.qpos[self.q], d.qpos[self.q] / (2 * math.pi)


def _registry():
    from .gpu_rush import GpuRushAdapter
    from .log_roll import LogRollAdapter
    from .race import RaceAdapter
    from .saucer import SaucerAdapter
    from .sumo import SumoAdapter
    from .wobbly_raft import WobblyRaftAdapter
    from .turntable import TurntableAdapter
    from .wheel import WheelAdapter
    return {"turntable": TurntableAdapter, "wheel": WheelAdapter, "saucer": SaucerAdapter, "sumo": SumoAdapter,
            # PvP wave 2 (kit 1.9.0, LB-108): the shipped wobbly-raft; the night's 1v1 prototypes (gym/CLI only).
            # The Goose Chase adapter is gone with its Challenges (2026-10-19, LB-109): asymmetric roles would need two models.
            "wobbly-raft": WobblyRaftAdapter, "log-roll": LogRollAdapter, "log-roll-swift": LogRollAdapter,
            # GPU Rush 1v1 (was Bread Fight v2 1v1, kit 1.10.0; renamed in 1.10.1; gpu-rush, four ducks, is refused by
            # duel.py)
            "gpu-rush-1v1": GpuRushAdapter,
            # arena races (DuelRules.finish): the first trunk into the finish zone wins
            "maze-race": RaceAdapter, "cliff": RaceAdapter}


def make_adapter(env):
    cls = _registry().get(env.info["challenge"]["id"])
    if cls is None:
        cls = HingedAdapter if env.info.get("props") else GenericAdapter
    return cls(env)


def adapter_columns(info: dict) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """(COLUMNS incl. inherited, ALIASES, DOC) of the Challenge's adapter class, without building one."""
    cls = _registry().get(info["challenge"]["id"]) or (HingedAdapter if info.get("props") else GenericAdapter)
    cols, aliases, doc = [], {}, {}
    for c in reversed(cls.__mro__):
        cols += [x for x in c.__dict__.get("COLUMNS", []) if x not in cols]
        aliases.update(c.__dict__.get("ALIASES", {}))
        doc.update(c.__dict__.get("DOC", {}))
    return cols, aliases, doc
