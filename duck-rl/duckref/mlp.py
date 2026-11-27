"""NumPy runner for the `duck-mlp/1` policy export (the format the TS MLP runner reads).

Format (written by tools/py/scripts/build_plant.py):
  <name>.mlp.bin   little-endian float32, all tensors concatenated.
  <name>.mlp.json  {format, source_onnx_sha256, input_dim, output_dim,
                    norm: {sub_offset, div_offset, len},
                    layers: [{in, out, weight_offset, bias_offset, weight_layout: "out_major",
                              activation: "elu" | "none", elu_alpha}]}
  Offsets count float32 elements. out_major means W[o][i] is at weight_offset + o*in + i.

Forward pass (mirrors the ONNX graph Sub -> Div -> (Gemm -> Elu) x3 -> Gemm):
  x = (obs - sub) / div                      elementwise, float32 (ONNX Sub/Div are float32)
  y = W x + b                                per layer, accumulated in float64, stored as float32
  y = y if y > 0 else alpha * (exp(y) - 1)   Elu on hidden layers only
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FORMAT = "duck-mlp/1"


class DuckMlp:
    def __init__(self, json_path: str | Path):
        json_path = Path(json_path)
        self.spec = json.loads(json_path.read_text())
        if self.spec["format"] != FORMAT:
            raise ValueError(f"{json_path}: format {self.spec['format']!r} != {FORMAT!r}")
        bin_path = json_path.with_name(json_path.name.removesuffix(".json") + ".bin")
        buf = np.fromfile(bin_path, dtype="<f4").astype(np.float32)

        norm = self.spec["norm"]
        self.sub = buf[norm["sub_offset"] : norm["sub_offset"] + norm["len"]]
        self.div = buf[norm["div_offset"] : norm["div_offset"] + norm["len"]]
        self.layers = []
        for L in self.spec["layers"]:
            assert L["weight_layout"] == "out_major"
            W = buf[L["weight_offset"] : L["weight_offset"] + L["out"] * L["in"]].reshape(L["out"], L["in"])
            b = buf[L["bias_offset"] : L["bias_offset"] + L["out"]]
            self.layers.append((W.astype(np.float64), b.astype(np.float64), L["activation"], L["elu_alpha"]))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """obs float32[..., input_dim] -> actions float32[..., output_dim]."""
        x = (np.asarray(obs, np.float32) - self.sub) / self.div
        for W, b, act, alpha in self.layers:
            y = (x.astype(np.float64) @ W.T + b).astype(np.float32)
            if act == "elu":
                y64 = y.astype(np.float64)
                y = np.where(y64 > 0.0, y64, alpha * np.expm1(y64)).astype(np.float32)
            x = y
        return x
