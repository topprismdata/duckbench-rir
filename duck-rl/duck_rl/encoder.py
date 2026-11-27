"""A frozen shared camera encoder in numpy (kit 1.27.0, train.py ENCODER): the site's camera-move CNN (packages/sim/src/policy/
cnn.ts, schema move-camera.ts duck-cam/1) on the ray picture in the shared palette ("ray-rgbd-palette-4x16x32/2", duck_rl/vision.py
RaySight.refresh(cheap=...)), then a linear named head and its standardisation: per picture, `out` named numbers
(the kit's encoder emb1: a class-free 64-number embedding, head = identity). The move reads those numbers after its 61 stock inputs; the encoder never trains. At export the head and the standardisation fold into layer 0's columns over the CNN's 32 features (fold_layer0),
so the site computes the same numbers with its own camera runner at the same cost.

Files (duck_rl/encoders/<name>/, written by tools/encoder_from_pt.py from the GPU trainer's .pt): encoder.json
(duck-cam-encoder/1: convs and dense exactly as a duck-cam/1 camera block, head, out_mu / out_sd offsets, names) and
encoder.bin (little-endian float32). The encoder's id is "enc_" + the first 12 hex digits of sha256(encoder.bin).

The forward pass follows cnn.ts: each layer sums in float64 and stores float32 (the ELU on the float32-rounded sum), so
the kit's features are the site's to float32 rounding. numpy (and one numba ELU kernel), batched over pictures [B, 4, 16, 32].
"""
from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from numba import njit

ENCODER_FORMAT = "duck-cam-encoder/1"
PICTURE = "ray-rgbd-palette-4x16x32/2"
# A retired picture (packages/schema move-camera.ts CAMERA_PICTURES_RETIRED, the same words): an encoder file that declares
# one is refused, as the CLI and the server refuse a move that declares it (the kit renders PICTURE only). The cheap look
# "ray-rgbd-cheap-4x16x32/1" (colours from the physics model) was retired on 2026-11-13 for the shared palette
# (docs/vision/PALETTE.md), and palette v1's picture "ray-rgbd-palette-4x16x32/1" on 2026-11-14 for palette v2; the kit
# ships only encoders trained on PICTURE.
RETIRED_PICTURES = {"ray-rgbd-cheap-4x16x32/1": 'the cheap look (colours from the physics model) was replaced by the shared palette on '
                    '2026-11-13; retrain the move on "ray-rgbd-palette-4x16x32/2" (a camera encoder for palette v2)',
                    "ray-rgbd-palette-4x16x32/1": 'palette v1\'s picture was replaced by palette v2 («Сумерки») on 2026-11-14; retrain the move on "ray-rgbd-palette-4x16x32/2" (a camera encoder for palette v2)'}
PICTURE_SHAPE = (4, 16, 32)
CAMERA_FORMAT = "duck-cam/1"
ENCODERS_DIR = Path(__file__).resolve().with_name("encoders")


def available() -> list[str]:
    """The encoders this kit ships (duck_rl/encoders/<name>/encoder.json)."""
    return sorted(p.parent.name for p in ENCODERS_DIR.glob("*/encoder.json"))


# The kit's encoder (Denis, 2026-11-16, LB-334): new camera moves train on it.
DEFAULT = "emb1"
# Encoders removed from the kit, with where their record lives (the unknown-encoder message adds the note).
REMOVED = {"v8.6": "removed from the kit on 2026-11-16 (Denis, LB-337); its record is "
                   "experiments/2026-11-13-robust-base-moves/final/encoder-v8.6"}


@lru_cache(maxsize=None)
def load(name: str) -> "Encoder":
    """The shipped encoder `name` (e.g. "emb1", the kit's encoder), once per process."""
    d = ENCODERS_DIR / name
    if not (d / "encoder.json").is_file():
        note = f"; {name} was {REMOVED[name]}" if name in REMOVED else ""
        raise ValueError(f"ENCODER {name!r}: no such encoder in this kit (it has: {', '.join(available()) or 'none'}){note}")
    return Encoder(d)


def _elu32(v: np.ndarray) -> np.ndarray:
    """cnn.ts: elu(Math.fround(sum)) stored float32, i.e. v > 0 ? v : Math.expm1(v) (so -0.0 stays -0.0). The reference for
    _elu64 (tools/encoder_test.py); latent uses _elu64. np.where on the input, not np.minimum: np.minimum(-0.0, 0.0) is
    -0.0 on macOS and +0.0 on Linux."""
    v = v.astype(np.float32).astype(np.float64)
    return np.where(v > 0, v, np.expm1(np.where(v > 0, 0.0, v))).astype(np.float32)


@njit(cache=True)
def _elu64_kernel(a: np.ndarray, out: np.ndarray) -> None:
    for i in range(a.size):
        x = np.float64(np.float32(a[i]))
        if not x > 0.0:  # x <= 0 or NaN, as _elu32's np.where(v > 0, ...)
            x = np.float64(np.float32(math.expm1(x)))  # cnn.ts: expm1(-0.0) is -0.0, expm1(NaN) NaN
        out[i] = x


def _elu64(v: np.ndarray) -> np.ndarray:
    """_elu32 in one numba pass, as float64 (the next layer's input): expm1 only on the non-positive sums. numpy's expm1
    has no SIMD path on Apple arm64 and _elu32 ran it on every activation (positives too) with four temporaries: most of
    the CNN's time on a Mac (PR #1, 2026-11-15). Bit-identical to _elu32 (tools/encoder_test.py checks every float32
    input below 0 against np.expm1)."""
    a = np.ascontiguousarray(v, np.float64)
    out = np.empty(a.shape, np.float64)
    _elu64_kernel(a.reshape(-1), out.reshape(-1))
    return out


class Encoder:
    def __init__(self, d: Path):
        d = Path(d)
        self.dir = d
        self.spec = s = json.loads((d / "encoder.json").read_text())
        if s.get("format") != ENCODER_FORMAT:
            raise ValueError(f"{d / 'encoder.json'}: format {s.get('format')!r}, not {ENCODER_FORMAT}")
        if s.get("picture") in RETIRED_PICTURES:
            raise ValueError(f"{d / 'encoder.json'}: picture \"{s['picture']}\" is retired: {RETIRED_PICTURES[s['picture']]}")
        if s.get("picture") != PICTURE:
            raise ValueError(f"{d / 'encoder.json'}: picture {s.get('picture')!r}; the kit renders {PICTURE}")
        self.blob = (d / "encoder.bin").read_bytes()
        self.id = f"enc_{hashlib.sha256(self.blob).hexdigest()[:12]}"
        self.name = s["name"]
        w = np.frombuffer(self.blob, "<f4")
        cut = lambda off, n: w[off:off + n].copy()  # noqa: E731
        self.convs = []
        c, h, wd = PICTURE_SHAPE
        self.macs = 0
        for v in s["convs"]:
            assert v["in"] == c and v["activation"] in ("elu", "none"), v
            k, st, p = v["k"], v["stride"], v["pad"]
            W = cut(v["weight_offset"], v["out"] * v["in"] * k * k).reshape(v["out"], v["in"] * k * k)
            self.convs.append((v, W.astype(np.float64), cut(v["bias_offset"], v["out"]).astype(np.float64)))
            h, wd, c = (h + 2 * p - k) // st + 1, (wd + 2 * p - k) // st + 1, v["out"]
            self.macs += c * h * wd * v["in"] * k * k
        dn = s["dense"]
        assert dn["in"] == c * h * wd, (dn["in"], c, h, wd)
        self.dense = (dn, cut(dn["weight_offset"], dn["out"] * dn["in"]).reshape(dn["out"], dn["in"]).astype(np.float64),
                      cut(dn["bias_offset"], dn["out"]).astype(np.float64))
        self.macs += dn["in"] * dn["out"]
        hd = s["head"]
        assert hd["in"] == dn["out"], (hd, dn)
        self.head_w = cut(hd["weight_offset"], hd["out"] * hd["in"]).reshape(hd["out"], hd["in"])
        self.head_b = cut(hd["bias_offset"], hd["out"])
        self.mu = cut(s["out_mu_offset"], hd["out"])
        self.sd = cut(s["out_sd_offset"], hd["out"])
        self.names = list(s["names"])
        self.latent_dim, self.out = dn["out"], hd["out"]
        assert len(self.names) == self.out

    # ------------------------------------------------------------------------------------------------ forward

    def latent(self, g: np.ndarray) -> np.ndarray:
        """Pictures [B, 4, 16, 32] (or one [4, 16, 32]) -> the CNN's features [B, latent_dim] float32, as cnn.ts CameraNet."""
        x = np.asarray(g, np.float32)
        one = x.ndim == 3
        x = x.reshape((-1,) + PICTURE_SHAPE).astype(np.float64)
        B = x.shape[0]
        for v, W, b in self.convs:
            k, st, p = v["k"], v["stride"], v["pad"]
            if p:
                x = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
            H, Wd = x.shape[2], x.shape[3]
            oh, ow = (H - k) // st + 1, (Wd - k) // st + 1
            if k == 1 and st == 1:
                cols = x.reshape(B, x.shape[1], H * Wd)
            else:  # [B, C, k*k, oh, ow] -> [B, C*k*k, oh*ow]: column i*k*k + ky*k + kx, PyTorch's W[o][i][ky][kx]
                cols = np.stack([x[:, :, ky:ky + st * (oh - 1) + 1:st, kx:kx + st * (ow - 1) + 1:st]
                                 for ky in range(k) for kx in range(k)], 2).reshape(B, x.shape[1] * k * k, oh * ow)
            y = np.matmul(W, cols) + b[:, None]
            y = _elu64(y) if v["activation"] == "elu" else y.astype(np.float32).astype(np.float64)
            x = y.reshape(B, v["out"], oh, ow)
        dn, Wd_, bd = self.dense
        f = x.reshape(B, -1) @ Wd_.T + bd
        f = _elu64(f).astype(np.float32) if dn["activation"] == "elu" else f.astype(np.float32)
        return f[0] if one else f

    def named(self, f: np.ndarray) -> np.ndarray:
        """CNN features [B, 32] -> the named numbers, standardised: (head(f) - mu) / sd (float32, the trainer's NamedCam)."""
        f = np.asarray(f, np.float32)
        return ((f @ self.head_w.T + self.head_b - self.mu) / self.sd).astype(np.float32)

    def features(self, g: np.ndarray) -> np.ndarray:
        """Pictures -> the `out` named numbers the move reads (standardised)."""
        return self.named(self.latent(g))

    # ------------------------------------------------------------------------------------------------ export

    def fold_layer0(self, Wn: np.ndarray, b0: np.ndarray, sub: np.ndarray | None = None, div: np.ndarray | None = None):
        """Layer 0's weights on the named numbers Wn [out, self.out] (over (z - sub) / div, the actor's normaliser on those
        slots; default 0 / 1) and its bias b0 -> (columns over the CNN's latent_dim features [out, latent_dim], new bias),
        float64: Wn ((H f + hb - mu) / sd - sub) / div = (Wn / (sd div)) H f + (Wn / div) ((hb - mu) / sd - sub)."""
        Wn = np.asarray(Wn, np.float64)
        sub = np.zeros(self.out) if sub is None else np.asarray(sub, np.float64).reshape(-1)
        div = np.ones(self.out) if div is None else np.asarray(div, np.float64).reshape(-1)
        Wd = Wn / div[None, :]
        sd, mu = self.sd.astype(np.float64), self.mu.astype(np.float64)
        cols = (Wd / sd[None, :]) @ self.head_w.astype(np.float64)
        bias = np.asarray(b0, np.float64) + Wd @ ((self.head_b.astype(np.float64) - mu) / sd - sub)
        return cols, bias

    def camera_block(self, offset: int) -> tuple[dict, list[np.ndarray]]:
        """The duck-cam/1 block (schema move-camera.ts) with the CNN's weights laid out from float `offset` of the move's
        .bin, and those float32 arrays in order."""
        w = np.frombuffer(self.blob, "<f4")
        parts, convs = [], []
        off = offset

        def put(src_off: int, n: int) -> int:
            nonlocal off
            parts.append(w[src_off:src_off + n].copy())
            at, off = off, off + n
            return at

        for v, _, _ in self.convs:
            n = v["out"] * v["in"] * v["k"] * v["k"]
            convs.append({"in": v["in"], "out": v["out"], "k": v["k"], "stride": v["stride"], "pad": v["pad"],
                          "weight_offset": put(v["weight_offset"], n), "bias_offset": put(v["bias_offset"], v["out"]),
                          "activation": v["activation"]})
        dn = self.dense[0]
        dense = {"in": dn["in"], "out": dn["out"], "weight_offset": put(dn["weight_offset"], dn["in"] * dn["out"]),
                 "bias_offset": put(dn["bias_offset"], dn["out"]), "activation": dn["activation"]}
        return {"format": CAMERA_FORMAT, "picture": PICTURE, "convs": convs, "dense": dense, "out": dn["out"]}, parts
