"""duck-sense/1 reference evaluator (numpy), the twin of packages/sim/src/runtime/sense-eval.ts. A verbatim copy of
spikes/senses-build/sense_ref.py (so a player's kit has it) except the palette, read from the kit's duck_rl/palette.json
(the same file as packages/schema/palette.json); tools/sense_test.py checks it on the repo's parity fixture.

Contract: docs/senses-design.md §5a and the header of packages/schema/src/sense.ts. Validation is the TS side's
(senseProblem); this file assumes a valid spec and only evaluates it. Feature order: [conv] + [body] + [cam] + [ops] +
(prev: previous body, cam, ops) + (prev_conv: previous conv features) + (prev_out: previous outputs); norm covers all after [conv].

    from sense_ref import Sense
    s = Sense.load("x.sense.json")          # reads x.sense.bin beside it
    out = s.step(picture, body, cam)        # picture: float32 [2048] or [4, 16, 32] (palette picture) or None

Numerics: the conv block stores float32 after each layer (as policy/cnn.ts; sums in float64), everything else is float64
over the float32 weights. Parity with TS: within 1e-6 on test/fixtures/sense (`python sense_ref.py --check <dir>`).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

PAL = json.loads((Path(__file__).resolve().parents[1] / "palette.json").read_text())

CAP = 3.0
ROWS, COLS = 16, 32
PLANE = ROWS * COLS
BODY_GROUPS = {"gyro": 3, "gravity": 3, "joint_pos": 14, "joint_vel": 14}
# The groups' slices of the 61 stock move inputs (sim obs.ts buildObs).
BODY_SLICES = {"gyro": (0, 3), "gravity": (3, 6), "joint_pos": (6, 20), "joint_vel": (20, 34)}

# colour -> class id (schema sense.ts pictureClassId)
SELF_MAX = 0.09
SKY_TOL = 0.003
CHROMA_NAMES = ["ground", "platform", "wall", "obstacle", "prop", "duck", "duck_accent"]
_COL = np.array([PAL["colours"][n] for n in CHROMA_NAMES], dtype=np.float64)
CHROMA = _COL / _COL.sum(1, keepdims=True)
CHROMA_IDS = np.array([PAL["ids"][n] for n in CHROMA_NAMES])
SKY = np.array(PAL["sky"], dtype=np.float64)


def class_ids(pic: np.ndarray) -> np.ndarray:
    """Class id per ray [512] of a palette picture [4, 512] (float)."""
    rgb = pic[1:4].T.astype(np.float64)  # [512, 3]
    s = np.maximum(rgb.sum(-1), 1e-6)
    ch = rgb / s[:, None]
    d = ((ch[:, None, :] - CHROMA[None]) ** 2).sum(-1)
    ids = CHROMA_IDS[d.argmin(-1)]
    ids = np.where(rgb.max(-1) <= SELF_MAX, PAL["ids"]["self"], ids)
    ids = np.where((np.abs(rgb - SKY[None]) <= SKY_TOL).all(-1), PAL["ids"]["sky"], ids)
    return ids


def value(x) -> float:
    """A vector value as a sense reads it: booleans 1 / 0, +-inf +-3, nan 0."""
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    x = float(x)
    if math.isnan(x):
        return 0.0
    if x == math.inf:
        return CAP
    if x == -math.inf:
        return -CAP
    return x


def none_of(o) -> float:
    n = o["none"]
    return math.inf if n == "Infinity" else -math.inf if n == "-Infinity" else float(n)


def _elu(v):
    return np.where(v > 0, v, np.expm1(np.minimum(v, 0)))


def _conv(x: np.ndarray, w: np.ndarray, b: np.ndarray, k: int, stride: int, pad: int, act: bool) -> np.ndarray:
    """PyTorch Conv2d (zero padding), float64 sums, stored float32 (ELU on the float32-rounded sum)."""
    cin, h, wd = x.shape
    oh = (h + 2 * pad - k) // stride + 1
    ow = (wd + 2 * pad - k) // stride + 1
    xp = np.zeros((cin, h + 2 * pad, wd + 2 * pad), dtype=np.float64)
    xp[:, pad:pad + h, pad:pad + wd] = x
    cols = np.empty((cin, k, k, oh, ow), dtype=np.float64)
    for ky in range(k):
        for kx in range(k):
            cols[:, ky, kx] = xp[:, ky:ky + stride * oh:stride, kx:kx + stride * ow:stride]
    v = np.einsum("oikl,iklyx->oyx", w.astype(np.float64), cols) + b.astype(np.float64)[:, None, None]
    return _finish(v, act)


def _finish(v: np.ndarray, act: bool) -> np.ndarray:
    v32 = v.astype(np.float32)
    return _elu(v32.astype(np.float64)).astype(np.float32) if act else v32


class Sense:
    def __init__(self, spec: dict, weights: np.ndarray):
        self.j = spec
        self.w = np.asarray(weights, dtype=np.float32)
        j = spec
        self.inputs = j["inputs"]
        self.ops = j.get("ops") or []
        self.outputs = j["outputs"]
        self.none = [none_of(o) for o in self.outputs]
        self.gate = j.get("gate")
        self.n_cur = sum(BODY_GROUPS.get(n, 1) for n in self.inputs["body"]) + len(self.inputs["cam"]) + len(self.ops)
        # inputs.prev_conv (absent = false): the previous tick's conv features, after the prev block, before prev_out
        self.n_prev_conv = j["conv"]["dense"]["out"] if self.inputs.get("prev_conv") else 0
        nv = self.n_cur * (2 if self.inputs["prev"] else 1) + self.n_prev_conv + (len(self.outputs) if self.inputs["prev_out"] else 0)
        self.mean = self._sl(j["norm"]["mean_offset"], nv).astype(np.float64)
        self.std = self._sl(j["norm"]["std_offset"], nv).astype(np.float64)
        assert np.all(self._sl(j["norm"]["std_offset"], nv) >= np.float32(1e-6)) and np.all(np.abs(self.w.astype(np.float64)) <= 1e6)  # SENSE_STD_MIN, SENSE_ABS_MAX
        self.reset()

    @classmethod
    def load(cls, json_path: str | Path) -> "Sense":
        p = Path(json_path)
        spec = json.loads(p.read_text())
        bin_path = p.with_name(p.name[: -len(".json")] + ".bin")
        return cls(spec, np.fromfile(bin_path, dtype="<f4"))

    def _sl(self, off: int, n: int) -> np.ndarray:
        return self.w[off:off + n]

    def reset(self) -> None:
        self.first = True
        self.prev_cur = np.zeros(self.n_cur)
        self.prev_conv = np.zeros(self.n_prev_conv)
        self.prev_out = np.zeros(len(self.outputs))

    def conv_features(self, pic: np.ndarray) -> np.ndarray:
        c = self.j["conv"]
        x = pic.reshape(4, ROWS, COLS).astype(np.float32)
        for v in c["convs"]:
            w = self._sl(v["weight_offset"], v["out"] * v["in"] * v["k"] * v["k"]).reshape(v["out"], v["in"], v["k"], v["k"])
            b = self._sl(v["bias_offset"], v["out"])
            x = _conv(x, w, b, v["k"], v["stride"], v["pad"], v["activation"] == "elu")
        d = c["dense"]
        w = self._sl(d["weight_offset"], d["out"] * d["in"]).reshape(d["out"], d["in"]).astype(np.float64)
        b = self._sl(d["bias_offset"], d["out"]).astype(np.float64)
        return _finish(w @ x.reshape(-1).astype(np.float64) + b, d["activation"] == "elu")

    def op_value(self, o: dict, pic: np.ndarray, cls: np.ndarray | None = None) -> float:
        """`cls`: the picture's class plane [512] (class_ids), decoded once per tick for all class_frac ops."""
        p = pic.reshape(4, ROWS, COLS)
        r0, r1 = o["rows"]
        c0, c1 = o["cols"]
        if o["op"] == "class_frac":
            plane = class_ids(pic.reshape(4, PLANE)) if cls is None else cls
            ids = plane.reshape(ROWS, COLS)[r0:r1 + 1, c0:c1 + 1]
            return float((ids == PAL["ids"][o["class"]]).mean())
        d = p[0, r0:r1 + 1, c0:c1 + 1].astype(np.float64) * CAP
        return float({"min": d.min, "max": d.max, "mean": d.mean}[o["op"]]())

    def step(self, picture, body: dict, cam: dict) -> np.ndarray:
        j, inp = self.j, self.inputs
        pic = None if picture is None else np.asarray(picture, dtype=np.float32).reshape(-1)
        feats = self.conv_features(pic).astype(np.float64) if "conv" in j else np.zeros(0)
        cur = []
        for n in inp["body"]:
            v = body[n]
            if n in BODY_GROUPS:
                assert len(v) == BODY_GROUPS[n]
                cur += [value(a) for a in v]
            else:
                cur.append(value(v))
        cur += [value(cam[n]) for n in inp["cam"]]
        cls = class_ids(pic.reshape(4, PLANE)) if any(o["op"] == "class_frac" for o in self.ops) else None
        cur += [self.op_value(o, pic, cls) for o in self.ops]
        cur = np.array(cur, dtype=np.float64)
        vec = [cur]
        if inp["prev"]:
            vec.append(cur.copy() if self.first else self.prev_cur)
        if self.n_prev_conv > 0:  # the previous tick's conv features (the first tick: the current ones)
            vec.append(feats.copy() if self.first else self.prev_conv)
            self.prev_conv = feats.copy()
        if inp["prev_out"]:
            vec.append(np.array([value(v) for v in (self.none if self.first else self.prev_out)], dtype=np.float64))
        self.prev_cur = cur.copy()
        v = np.concatenate(vec) if vec else np.zeros(0)
        v = (v - self.mean) / self.std
        h = np.concatenate([feats, v])
        for l in j["mlp"]:
            w = self._sl(l["weight_offset"], l["in"] * l["out"]).reshape(l["out"], l["in"]).astype(np.float64)
            b = self._sl(l["bias_offset"], l["out"]).astype(np.float64)
            z = w @ h + b
            a = l["activation"]
            h = np.tanh(z) if a == "tanh" else _elu(z) if a == "elu" else np.where(z > 0, z, 0.0) if a == "relu" else z
        out = np.empty(len(self.outputs))
        closed = False
        for i, o in enumerate(self.outputs):
            a, z = o["activation"], float(h[i])
            if a == "tanh":
                y = math.tanh(z)
            elif a == "sigmoid":
                y = 1 / (1 + math.exp(-z)) if z >= -709 else 0.0 if math.isfinite(z) or z < 0 else math.nan
            else:
                y = z
            if math.isfinite(y):  # a non-finite output reads its none, and a non-finite gate reads closed
                out[i] = min(o["max"], max(o["min"], y))
            else:
                out[i] = self.none[i]
                closed = closed or i == self.gate
        g = self.gate
        if g is not None and (closed or out[g] < 0.5):
            for i in range(len(out)):
                if i != g:
                    out[i] = self.none[i]
        self.prev_out = out.copy()
        self.first = False
        return out


def _num(x):
    """JSON fixture numbers: "Infinity" / "-Infinity" / "NaN" strings back to floats."""
    if isinstance(x, str):
        return float(x.replace("Infinity", "inf"))
    if isinstance(x, list):
        return [_num(a) for a in x]
    return x


def run_fixture(d: Path) -> list[list[float]]:
    """Every tick's outputs of the fixture in `d` (sense.sense.json/.bin, inputs.json, pictures.bin)."""
    s = Sense.load(d / "sense.sense.json")
    inp = json.loads((d / "inputs.json").read_text())
    ticks = inp["ticks"]
    pics = np.fromfile(d / inp["pictures"], dtype="<f4").reshape(len(ticks), -1)
    outs = []
    for t, pic in zip(ticks, pics, strict=True):
        body = {k: _num(v) for k, v in t["body"].items()}
        cam = {k: (v if isinstance(v, bool) else _num(v)) for k, v in t["cam"].items()}
        outs.append([float(v) for v in s.step(pic, body, cam)])
    return outs


def check(d: Path, tol: float = 1e-6) -> float:
    want = json.loads((d / "expected.json").read_text())["outputs"]
    got = run_fixture(d)
    worst = 0.0
    for g, w in zip(got, want, strict=True):
        for a, b in zip(g, _num(w), strict=True):
            if math.isinf(a) or math.isinf(b):
                assert a == b, (a, b)
            else:
                worst = max(worst, abs(a - b))
    assert worst <= tol, worst
    return worst


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        print(f"max |diff| {check(Path(sys.argv[2])):.3g}: OK")
    else:
        print(__doc__)
