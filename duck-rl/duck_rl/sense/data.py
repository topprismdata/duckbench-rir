"""A `duck-sense-collect/1` directory (`duckbench sense collect <challenge> <strategy> --seeds a-b --out dir`), read for
training: one row per duck per program tick (10 Hz).

  meta.json    {format, challenge, strategy_id, picture, seeds, n_rows, body_fields, cam_fields, world_fields}
  picture.f32  n_rows x 4 x 16 x 32 float32 LE (the palette picture the sense would see)
  body.f32     n_rows x len(body_fields) float32 (groups gyro 3, gravity 3, joint_pos 14, joint_vel 14, then scalars)
  cam.f32      n_rows x len(cam_fields) float32
  rows.jsonl   {seed, duck, tick, t} per row
  world.jsonl  a flat dict per row: every signal (privileged too), the true trunk pose / velocity, other ducks' poses,
               the result on the last row

`Data` concatenates one or more directories; `Row` is what a label function gets (`row.world`, `row.at(dt)`, `row.prev`).
Values reach a sense as sense-eval.ts reads them: booleans 1 / 0, +-Infinity +-3, NaN 0 (`vector()`).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

FORMAT = "duck-sense-collect/1"
PICTURE_SHAPE = (4, 16, 32)
PICTURE_SIZE = 4 * 16 * 32
CAP = 3.0
BODY_GROUPS = {"gyro": 3, "gravity": 3, "joint_pos": 14, "joint_vel": 14}
_GROUP_COL = re.compile(r"^(gyro|gravity|joint_pos|joint_vel)(?:\[(\d+)\]|[._](\d+))$")


def sense_values(a: np.ndarray) -> np.ndarray:
    """float64 copy with +-inf -> +-3 and NaN -> 0 (booleans are stored 1 / 0 already)."""
    a = np.asarray(a, np.float64).copy()
    a[np.isnan(a)] = 0.0
    a[a == math.inf] = CAP
    a[a == -math.inf] = -CAP
    return a


def _body_columns(fields: list[str], width: int) -> dict[str, list[int]]:
    """Input name -> its columns in body.f32. Accepts expanded group columns (gyro_0, gyro.0, gyro[0]) or a bare group
    name standing for its whole width."""
    cols: dict[str, list[int]] = {}
    at = 0
    for f in fields:
        m = _GROUP_COL.match(f)
        if m:
            cols.setdefault(m.group(1), []).append(at)
            at += 1
        elif f in BODY_GROUPS and width != len(fields):  # unexpanded: the group's whole width
            cols[f] = list(range(at, at + BODY_GROUPS[f]))
            at += BODY_GROUPS[f]
        elif f in BODY_GROUPS:  # repeated bare name, one column each
            cols.setdefault(f, []).append(at)
            at += 1
        else:
            cols[f] = [at]
            at += 1
    if at != width:
        raise ValueError(f"body_fields describe {at} columns; body.f32 has {width}")
    for g, n in BODY_GROUPS.items():
        if g in cols and len(cols[g]) != n:
            raise ValueError(f"body group {g} has {len(cols[g])} columns, want {n}")
    return cols


class Dir:
    """One collect directory."""

    def __init__(self, path: str | Path):
        self.path = p = Path(path).expanduser().resolve()
        self.meta = m = json.loads((p / "meta.json").read_text())
        if m.get("format") != FORMAT:
            raise ValueError(f"{p}/meta.json: format {m.get('format')!r}, not {FORMAT}")
        self.n = n = int(m["n_rows"])
        ch = m.get("challenge", p.name)
        self.challenge = str(ch["id"]) if isinstance(ch, dict) and "id" in ch else str(ch)  # the Challenge id (meta: {id, version})
        self.challenge_version = ch.get("version") if isinstance(ch, dict) else None
        self.picture_id = m.get("picture")
        self.camera_world = m.get("camera_world")  # CLI with kit 1.28.0 and later; None in older collections
        self.sim_version = m.get("sim_version")
        self.cam_fields = list(m["cam_fields"])
        body = np.fromfile(p / "body.f32", "<f4")
        if n and body.size % n:
            raise ValueError(f"{p}/body.f32: {body.size} floats is not a multiple of n_rows {n}")
        self.body = body.reshape(n, -1) if n else body.reshape(0, len(m["body_fields"]))
        self.body_cols = _body_columns(list(m["body_fields"]), self.body.shape[1])
        cam = np.fromfile(p / "cam.f32", "<f4")
        self.cam = cam.reshape(n, len(self.cam_fields))
        pic = p / "picture.f32"
        self.picture = np.memmap(pic, "<f4", "r", shape=(n, *PICTURE_SHAPE)) if pic.exists() and pic.stat().st_size else None
        if self.picture is not None and pic.stat().st_size != n * PICTURE_SIZE * 4:
            raise ValueError(f"{pic}: {pic.stat().st_size} bytes, want n_rows x 2048 float32")
        rows = [json.loads(s) for s in (p / "rows.jsonl").read_text().splitlines() if s.strip()]
        world = [json.loads(s) for s in (p / "world.jsonl").read_text().splitlines() if s.strip()]
        if len(rows) != n or len(world) != n:
            raise ValueError(f"{p}: rows.jsonl {len(rows)} / world.jsonl {len(world)} lines, n_rows {n}")
        self.seed = np.array([r["seed"] for r in rows], np.int64)
        self.duck = np.array([r["duck"] for r in rows], np.int64)
        self.tick = np.array([r["tick"] for r in rows], np.int64)
        self.t = np.array([r["t"] for r in rows], np.float64)
        self.world = world

    def collect_command(self) -> str:
        m = self.meta
        if isinstance(m.get("command"), str):
            return m["command"]
        s = m.get("seeds")
        if isinstance(s, list) and s and all(isinstance(x, int) for x in s):
            s = f"{min(s)}-{max(s)}" if sorted(s) == list(range(min(s), max(s) + 1)) else ",".join(map(str, s))
        ch = self.challenge if self.challenge_version is None else f"{self.challenge}@{self.challenge_version}"
        return f"duckbench sense collect {ch} {m.get('strategy_id')} --seeds {s} --out {self.path.name}"


class Data:
    """One or more collect directories as one table. Global row i; `seq` groups rows of one (dir, seed, duck)."""

    def __init__(self, dirs: list[str | Path]):
        if not dirs:
            raise ValueError("no data directories")
        self.dirs = [Dir(d) for d in dirs]
        self.n = sum(d.n for d in self.dirs)
        self.dir_of = np.concatenate([np.full(d.n, k, np.int64) for k, d in enumerate(self.dirs)])
        self.local = np.concatenate([np.arange(d.n) for d in self.dirs])
        self.seed = np.concatenate([d.seed for d in self.dirs])
        self.duck = np.concatenate([d.duck for d in self.dirs])
        self.tick = np.concatenate([d.tick for d in self.dirs])
        self.t = np.concatenate([d.t for d in self.dirs])
        self.world = [w for d in self.dirs for w in d.world]
        # sequences: rows of one (dir, seed, duck) sorted by tick; prev / next by position
        key = np.stack([self.dir_of, self.seed, self.duck], 1)
        order = np.lexsort((self.tick, self.duck, self.seed, self.dir_of))
        self.prev_idx = np.full(self.n, -1, np.int64)
        self.seq_of = np.zeros(self.n, np.int64)
        self.pos = np.zeros(self.n, np.int64)
        self.seqs: list[np.ndarray] = []
        ks = key[order]
        cuts = np.flatnonzero(np.any(ks[1:] != ks[:-1], 1)) + 1 if self.n else np.zeros(0, np.int64)
        for s in np.split(order, cuts) if self.n else []:
            self.seq_of[s] = len(self.seqs)
            self.pos[s] = np.arange(len(s))
            self.prev_idx[s[1:]] = s[:-1]
            self.seqs.append(s)
        self.seq_t = [self.t[s] for s in self.seqs]

    # ------------------------------------------------------------------------------------------------ rows

    def row(self, i: int) -> "Row":
        return Row(self, int(i))

    def at(self, i: int, dt: float) -> int:
        """The same duck's row dt seconds after row i in the same seed (the first at or after t + dt), or -1."""
        q = self.seq_of[i]
        st = self.seq_t[q]
        j = int(np.searchsorted(st, self.t[i] + dt - 1e-6))
        if j >= len(st) or (dt < 0 and j > self.pos[i]):
            return -1
        return int(self.seqs[q][j])

    def challenge_of(self, i: int) -> str:
        return self.dirs[self.dir_of[i]].challenge

    def group_names(self) -> list[str]:
        """Per-world names for the eval: the Challenge, or the directory name when two dirs share one."""
        ch = [d.challenge for d in self.dirs]
        return [c if ch.count(c) == 1 else f"{c}:{d.path.name}" for c, d in zip(ch, self.dirs)]

    # ------------------------------------------------------------------------------------------------ features

    def body_names(self) -> list[str]:
        names = set(self.dirs[0].body_cols)
        for d in self.dirs[1:]:
            names &= set(d.body_cols)
        return sorted(names)

    def cam_names(self) -> list[str]:
        names = set(self.dirs[0].cam_fields)
        for d in self.dirs[1:]:
            names &= set(d.cam_fields)
        return [n for n in self.dirs[0].cam_fields if n in names]

    def current(self, body: list[str], cam: list[str]) -> np.ndarray:
        """The current tick's body + cam values per row [n, width] float64, as a sense reads them."""
        parts = []
        for d in self.dirs:
            cols = []
            for b in body:
                if b not in d.body_cols:
                    raise ValueError(f"{d.path}: no body input {b!r} (has {', '.join(d.body_cols)})")
                cols += d.body_cols[b]
            ci = []
            for c in cam:
                if c not in d.cam_fields:
                    raise ValueError(f"{d.path}: no cam input {c!r} (has {', '.join(d.cam_fields)})")
                ci.append(d.cam_fields.index(c))
            parts.append(np.concatenate([d.body[:, cols], d.cam[:, ci]], 1))
        return sense_values(np.concatenate(parts, 0))

    def vector(self, body: list[str], cam: list[str], prev: bool) -> np.ndarray:
        """The sense's vector (before norm): current + (prev ? the previous row's current, the first row its own)."""
        cur = self.current(body, cam)
        if not prev:
            return cur
        pi = np.where(self.prev_idx >= 0, self.prev_idx, np.arange(self.n))
        return np.concatenate([cur, cur[pi]], 1)

    def pictures(self, idx: np.ndarray) -> np.ndarray:
        """Pictures of rows idx [len, 4, 16, 32] float32."""
        idx = np.asarray(idx, np.int64)
        out = np.empty((len(idx), *PICTURE_SHAPE), np.float32)
        for k, d in enumerate(self.dirs):
            m = self.dir_of[idx] == k
            if m.any():
                if d.picture is None:
                    raise ValueError(f"{d.path}: no picture.f32 (a picture template needs it)")
                out[m] = d.picture[self.local[idx[m]]]
        return out


class Row:
    """What a label function gets for one row."""

    __slots__ = ("_d", "i")

    def __init__(self, d: Data, i: int):
        self._d, self.i = d, i

    @property
    def world(self) -> dict:
        return self._d.world[self.i]

    @property
    def seed(self) -> int:
        return int(self._d.seed[self.i])

    @property
    def duck(self) -> int:
        return int(self._d.duck[self.i])

    @property
    def tick(self) -> int:
        return int(self._d.tick[self.i])

    @property
    def t(self) -> float:
        return float(self._d.t[self.i])

    @property
    def challenge(self) -> str:
        return self._d.challenge_of(self.i)

    @property
    def prev(self) -> "Row | None":
        j = self._d.prev_idx[self.i]
        return None if j < 0 else Row(self._d, int(j))

    def at(self, dt: float) -> "Row | None":
        """The same duck's row dt seconds later in the same seed (None past the end)."""
        j = self._d.at(self.i, dt)
        return None if j < 0 else Row(self._d, j)

    def __repr__(self) -> str:
        return f"Row(seed={self.seed}, duck={self.duck}, tick={self.tick})"
