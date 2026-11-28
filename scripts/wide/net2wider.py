#!/usr/bin/env python3
"""Net2Wider for duck-mlp/1: double the first hidden layer, preserving the function.

  python scripts/wide/net2wider.py <in.mlp.json> <out.mlp.json> [--factor 2]

Rules (Chen et al. 2016, function-preserving):
  W1' = [W1; W1]          (rows = neurons duplicated)
  b1' = [b1; b1]
  W2' = [W2/f, W2/f]      (f = duplication factor, so W2' h1' == W2 h1)
Other layers copied unchanged. Verified on random observations in numpy.
"""
import argparse, json
from pathlib import Path

import numpy as np


def load(path):
    meta = json.loads(Path(path).read_text())
    blob = np.fromfile(str(path).replace(".json", ".bin"), dtype="<f4")
    norm = meta["norm"]
    sub = blob[norm["sub_offset"]:norm["sub_offset"] + norm["len"]].copy()
    div = blob[norm["div_offset"]:norm["div_offset"] + norm["len"]].copy()
    layers = []
    for L in meta["layers"]:
        w = blob[L["weight_offset"]:L["weight_offset"] + L["in"] * L["out"]].reshape(L["out"], L["in"])
        b = blob[L["bias_offset"]:L["bias_offset"] + L["out"]]
        layers.append((w.copy(), b.copy(), L["activation"], L.get("elu_alpha", 1.0)))
    return meta, sub, div, layers


def widen(layers, factor=2, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    w1, b1, act1, a1 = layers[0]
    w2, b2, act2, a2 = layers[1]
    w1n = np.tile(w1, (factor, 1))
    b1n = np.tile(b1, factor)
    if noise:  # break replica symmetry (Chen et al. break ties with a perturbation)
        n = w1.shape[0]
        for r in range(1, factor):
            w1n[r * n:(r + 1) * n] += rng.normal(scale=noise, size=w1.shape)
            b1n[r * n:(r + 1) * n] += rng.normal(scale=noise, size=w1.shape[0])
    w2n = np.hstack([w2 / factor] * factor)
    out = [(w1n, b1n, act1, a1), (w2n, b2, act2, a2)] + layers[2:]
    return out


def forward(x, sub, div, layers):
    h = (np.asarray(x, dtype=np.float32) - sub) / div
    for i, (w, b, act, alpha) in enumerate(layers):
        h = w @ h + b
        if act == "elu":
            h = np.where(h > 0, h, alpha * np.expm1(h))
        elif act == "tanh":
            h = np.tanh(h)
        elif act == "relu":
            h = np.maximum(h, 0)
    return h


def save(meta, sub, div, layers, out_path):
    out_path = Path(out_path)
    blob = []
    blob.append(sub.astype("<f4"))
    blob.append(div.astype("<f4"))
    layers_meta = []
    for w, b, act, alpha in layers:
        off_w = sum(len(x) for x in blob)
        blob.append(w.astype("<f4").reshape(-1))
        off_b = sum(len(x) for x in blob)
        blob.append(b.astype("<f4").reshape(-1))
        layers_meta.append({"in": int(w.shape[1]), "out": int(w.shape[0]), "weight_offset": int(off_w),
                            "bias_offset": int(off_b), "weight_layout": "out_major",
                            "activation": act, "elu_alpha": (alpha if act == "elu" else None)})
    m = dict(meta)
    m["layers"] = layers_meta
    m["norm"] = {"sub_offset": 0, "div_offset": len(sub), "len": len(sub)}
    m["meta"] = dict(meta.get("meta") or {})
    m["meta"]["net2wider"] = f"first hidden x{int(layers_meta[0]['out'] / meta['layers'][0]['out'])}"
    out_path.write_text(json.dumps(m, indent=1))
    with open(str(out_path).replace(".json", ".bin"), "wb") as f:
        f.write(np.concatenate(blob).astype("<f4").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--noise", type=float, default=0.0)
    a = ap.parse_args()
    meta, sub, div, layers = load(a.src)
    wide = widen(layers, a.factor, a.noise)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, sub.size)).astype(np.float32) * 0.5
    y0 = np.stack([forward(x, sub, div, layers) for x in X])
    y1 = np.stack([forward(x, sub, div, wide) for x in X])
    err = float(np.abs(y0 - y1).max())
    if a.noise == 0.0:
        assert err < 1e-4, f"function not preserved: {err}"
    else:
        print(f"[noise={a.noise}] forward delta {err:.3e} (expected ~1e-3, symmetry broken)")
    save(meta, sub, div, wide, a.dst)
    print(f"widened {a.src} -> {a.dst}: max|Δoutput|={err:.3e} on 64 random obs")


if __name__ == "__main__":
    main()