"""The TS sim's policy runner in Python, bit for bit: packages/sim/src/policy/mlp-scalar.ts (the plain-JS twin of the
WASM SIMD kernel MlpSimdRunner, the runner every episode and match uses) as a numba kernel.

duckref DuckMlp (tools/py/duckref/mlp.py) accumulates each layer in float64 (the ONNX graph's numbers); the sim adds in
float32: per output, 16 partial sums (lane l = the inputs i with i mod 16 = l, over the row zero-padded to a multiple of
16, in input order), then ((a0 + a1) + (a2 + a3)) lane-wise, ((l0 + l1) + (l2 + l3)), + bias. The two differ by ~1e-7
per action, which a contact-rich episode amplifies; SimdMlp does the sim's float32 operations in its order, so a Python
match (duck_rl/duel.py) feeds the physics the actions TS computes. The kernel keeps each lane sum's order but walks the
weights input-major (every output's lane sum for input i in one contiguous loop, which LLVM vectorises; ~4x faster
than the output-major loop and the same float32 operations). Normaliser: x = f32(f32(obs - sub) / div), as the JS
(float64 then rounded: the same as float32 arithmetic for - and /). ELU on hidden layers: v <= 0 -> alpha * expm1(v) in
float64, stored float32 (libm's expm1 vs V8's may differ in the last float64 bit; rounding to float32 hides it).
Checked bit for bit against ts/check_mlp.ts --only simd (tools/duel_parity.py --mlp-check).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numba import njit

from duckref.mlp import DuckMlp  # the duck-mlp/1 loader (format checks, offsets)

LANES = 16  # mlp-simd.wat consumes 16 floats per iteration


@njit(cache=True)
def _dense(wt, b, x, y, acc, n, n_out):
    """mlp-scalar.ts dense, y[o] = b[o] + W[o] . x in float32, with wt input-major (wt[i n_out + o] = W[o][i]):
    acc[l n_out + o] is lane l of output o, accumulated over i = l, l + 16, ... in order; n a multiple of 16. (wt is
    the layer's own slice: an offset in the index stops LLVM from vectorising the inner loop, 5x slower.)"""
    for j in range(LANES * n_out):
        acc[j] = np.float32(0.0)
    for i in range(n):
        xv = x[i]
        a = (i % LANES) * n_out
        r = i * n_out
        for o in range(n_out):
            acc[a + o] += wt[r + o] * xv
    for o in range(n_out):
        l0 = (acc[o] + acc[4 * n_out + o]) + (acc[8 * n_out + o] + acc[12 * n_out + o])
        l1 = (acc[n_out + o] + acc[5 * n_out + o]) + (acc[9 * n_out + o] + acc[13 * n_out + o])
        l2 = (acc[2 * n_out + o] + acc[6 * n_out + o]) + (acc[10 * n_out + o] + acc[14 * n_out + o])
        l3 = (acc[3 * n_out + o] + acc[7 * n_out + o]) + (acc[11 * n_out + o] + acc[15 * n_out + o])
        y[o] = ((l0 + l1) + (l2 + l3)) + b[o]


@njit(cache=True)
def _forward(obs, sub, div, wt, b, woff, boff, npad, nout, elu, alpha, x0, buf0, buf1, acc, out):
    for i in range(sub.shape[0]):
        x0[i] = (obs[i] - sub[i]) / div[i]
    x = x0
    y = buf0
    which = 0
    for k in range(npad.shape[0]):
        y = buf0 if which == 0 else buf1
        n, m = npad[k], nout[k]
        _dense(wt[woff[k]:woff[k] + n * m], b[boff[k]:boff[k] + m], x, y, acc, n, m)
        if elu[k]:
            for o in range(nout[k]):
                v = np.float64(y[o])
                if v <= 0.0:
                    y[o] = np.float32(alpha[k] * math.expm1(v))
        for o in range(nout[k], y.shape[0]):
            y[o] = np.float32(0.0)  # zero padding: the next layer reads it
        x = y
        which ^= 1
    for o in range(out.shape[0]):
        out[o] = y[o]


class SimdMlp:
    """A duck-mlp/1 policy (x.mlp.json, x.mlp.bin beside it) run as the sim's MlpSimdRunner: run(obs, out) / __call__(obs).
    obs float32[input_dim] -> action float32[output_dim]; one observation at a time (the match runtime's use)."""

    def __init__(self, json_path: str | Path):
        ref = DuckMlp(json_path)
        self.path = Path(json_path)
        self.spec = ref.spec
        self.sub = np.ascontiguousarray(ref.sub, np.float32)
        self.div = np.ascontiguousarray(ref.div, np.float32)
        ws, bs, woff, boff, npad, nout, elu, alpha = [], [], [], [], [], [], [], []
        wo = bo = 0
        for (W, bias, act, a), L in zip(ref.layers, self.spec["layers"]):
            n_in, n_out = int(L["in"]), int(L["out"])
            pad = -(-n_in // LANES) * LANES
            Wt = np.zeros((pad, n_out), np.float32)
            Wt[:n_in] = W.T.astype(np.float32)  # DuckMlp widened the float32 blob to float64: exact back
            ws.append(Wt.ravel())
            bs.append(bias.astype(np.float32))
            woff.append(wo)
            boff.append(bo)
            wo += Wt.size
            bo += n_out
            npad.append(pad)
            nout.append(n_out)
            elu.append(act == "elu")
            alpha.append(float(a) if a is not None else 1.0)  # parseMlp: elu_alpha ?? 1
        self.input_dim, self.output_dim = int(self.spec["input_dim"]), int(self.spec["output_dim"])
        if self.sub.shape[0] != self.input_dim or nout[-1] != self.output_dim or any(
                npad[k] != -(-nout[k - 1] // LANES) * LANES for k in range(1, len(npad))):
            raise ValueError(f"{json_path}: inconsistent layer shapes")
        self.wt, self.b = np.concatenate(ws), np.concatenate(bs)
        self.woff, self.boff = np.asarray(woff, np.int64), np.asarray(boff, np.int64)
        self.npad, self.nout = np.asarray(npad, np.int64), np.asarray(nout, np.int64)
        self.elu, self.alpha = np.asarray(elu, np.bool_), np.asarray(alpha, np.float64)
        width = max(-(-n // LANES) * LANES for n in nout)
        self._x0 = np.zeros(npad[0], np.float32)
        self._buf0, self._buf1 = np.zeros(width, np.float32), np.zeros(width, np.float32)
        self._acc = np.zeros(LANES * max(nout), np.float32)

    def run(self, obs: np.ndarray, out: np.ndarray) -> np.ndarray:
        """obs float32[input_dim] (contiguous) -> `out` float32[output_dim] (returned)."""
        _forward(obs, self.sub, self.div, self.wt, self.b, self.woff, self.boff, self.npad, self.nout, self.elu,
                 self.alpha, self._x0, self._buf0, self._buf1, self._acc, out)
        return out

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.run(np.ascontiguousarray(obs, np.float32), np.empty(self.output_dim, np.float32))
