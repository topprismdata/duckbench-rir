"""Any 61 -> 14 MLP policy (or an encoder move: kit 1.27.0) -> duck-mlp/1 (the TS sim's policy format: <name>.mlp.json + little-endian float32 <name>.mlp.bin).

CLI:  to_mlp.py <input> <out.mlp.json> [--check] [--activation elu|none] [--eps auto] [--ref other.onnx] [--n 256]
      writes out.mlp.json + out.mlp.bin, validates (always) and, with --check, also runs the TS runners
      (ts/check_mlp.ts through duck_rl/bridge.py, the CLI's `kit check-mlp` in a player's kit: parseMlp +
      MlpSimdRunner as eval.ts, plus MlpRunner/MlpScalarRunner).

API:  load_any(path, activation="elu", eps=None) -> (sub, div, layers, activations)
      write_mlp(sub, div, layers, out_path, activation="elu", meta=None) -> spec dict
      mlp_forward(out_path, x) -> numpy float32 forward of the written blob
      convert(src, out_path, check=False, ...) -> report dict (load + write + validate [+ TS check])
  sub, div: float32[61] (out = (obs - sub) / div); layers: [(W[out, in], b[out]), ...];
  activation / activations: per layer "elu" | "none" | ("elu", alpha); a plain str "elu"/"none" means that activation
  after every layer but the last, the last has none.

Accepted inputs:
  (a) a move-lab checkpoint .pt (lab/learn.py save_ckpt): dict with "plain" = PlainActor.state_dict()
      {sub, div, net.0.weight, net.0.bias, net.2.weight, ...}, ELU between the Linear layers. If only "actor" is
      present (rsl_rl MLPModel state dict: mlp.{i}.weight/bias, obs_normalizer._mean/_std), sub = _mean,
      div = _std + eps (--eps; default auto: 0 for a move-lab checkpoint, which has
      "warm_start" and sets nz.eps = 0 in learn.py warm_start; else rsl_rl EmpiricalNormalization's 1e-2).
  (b) a .pt that is
      - a dict {"sub"?, "div"?, "layers": [(W, b), ...], "activation"?: str | [per layer]} (W as [out, in]);
      - a plain state dict / nn.Sequential state dict: every 2-D "<prefix><k>.weight" with its "<prefix><k>.bias",
        ordered by k (Linear layers; ELU/none modules carry no weights, so the activation between layers comes from
        --activation, default elu); normaliser from "sub"/"div" (any prefix), or "obs_normalizer._mean/_std" (+eps),
        or none (sub 0, div 1);
      - an nn.Module object (its state_dict, as above; activations read from its nn.ELU/nn.Identity modules when it is
        an nn.Sequential or has one).
  (c) an ONNX graph that is a chain from the single input to the single output: optional Sub(x - c), optional
      Div(x / c) or Mul(x * c) (div = 1/c), then layers Gemm | MatMul [+ Add(bias)], with optional Elu after each
      (no other activation: the TS runner supports only elu/none). Constant nodes, Identity nodes and constant-only
      subgraphs (Add/Sub/Mul/Div/Reshape/Unsqueeze/Squeeze/Cast/Sqrt/Transpose) are folded first, as torch.onnx.export
      leaves them. A missing normaliser = sub 0, div 1.
Encoder moves (kit 1.27.0, train.py ENCODER; --encoder NAME, or encoder= an encoder.py Encoder): a policy over the 61 then
the frozen encoder's `out` named numbers (61 + 64 for emb1) is written as the site's camera move, "61 + camera" (LB-330: until
then a 73-input eye move whose 12 eye columns were 0): its 61 inputs, a duck-cam/1 "camera" block (the encoder's convs and
dense layer, picture "ray-rgbd-palette-4x16x32/2") and layer 0's camera columns = the policy's weights on the named numbers
folded through the encoder's head and standardisation (Encoder.fold_layer0). Validated on random pictures: the
written file (its own CNN, read from the blob) vs the source on [obs61, the encoder's numbers of the same pictures].
Refused: non-finite weights, a zero in div, input not 61 (or 61 + the encoder's out), output != 14, chain breaks,
unsupported ops.
Validation: numpy forward of the written blob vs the source forward (torch for .pt, onnxruntime for .onnx; for a
checkpoint also its sibling ckpt-*.onnx and the "actor" branch if present) on n random obs (sub + 0.5 * normal).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
IN_DIM, OUT_DIM = 61, 14


class ConvertError(ValueError):
    pass


def _check(cond, msg):
    if not cond:
        raise ConvertError(msg)


def _np(t) -> np.ndarray:
    if hasattr(t, "detach"):
        t = t.detach().cpu().double().numpy()
    return np.asarray(t, dtype=np.float64)


def _acts(activation, n: int) -> list[tuple[str, float | None]]:
    """Normalise an activation spec to [(kind, alpha)] per layer."""
    def one(a):
        if isinstance(a, (tuple, list)):
            if a[0] == "elu":
                return ("elu", 1.0 if a[1] is None else float(a[1]))
            a = a[0]
        a = str(a).lower()
        if a == "elu":
            return ("elu", 1.0)
        if a in ("none", "identity", "linear"):
            return ("none", None)
        raise ConvertError(f"activation {a!r}: the TS runner supports only elu and none")
    if isinstance(activation, str):
        return [one(activation) for _ in range(n - 1)] + [("none", None)]
    acts = [one(a) for a in activation]
    _check(len(acts) == n, f"{len(acts)} activations for {n} layers")
    return acts


# ------------------------------------------------------------------------------------------------ writing

def fold_encoder(sub, div, layers, encoder) -> tuple:
    """A policy over [obs61, the encoder's `out` named numbers] -> the camera move's (sub, div, layers) over the 61 inputs
    then the encoder's latent_dim features: layer 0's columns = [its 61 body columns, the named columns folded through the
    encoder's head and standardisation (and the policy's normaliser on those slots)]. float64."""
    W0, b0 = layers[0]
    n_named = encoder.out
    _check(W0.shape[1] == IN_DIM + n_named, f"an encoder move's layer 0 reads {IN_DIM} + {n_named} inputs, got {W0.shape[1]}")
    sub = np.zeros(IN_DIM + n_named) if sub is None else _np(sub).reshape(-1)
    div = np.ones(IN_DIM + n_named) if div is None else _np(div).reshape(-1)
    cols, bias = encoder.fold_layer0(W0[:, IN_DIM:], b0, sub[IN_DIM:], div[IN_DIM:])
    W = np.concatenate([W0[:, :IN_DIM], cols], 1)
    return sub[:IN_DIM], div[:IN_DIM], [(W, bias)] + list(layers[1:])


def write_mlp(sub, div, layers, out_path, activation="elu", meta: dict | None = None, source_sha256: str = "",
              encoder=None) -> dict:
    """Write <stem>.mlp.json + <stem>.mlp.bin (offsets in floats, weights out-major). Returns the spec. With `encoder`
    (duck_rl/encoder.py Encoder): the policy reads [obs61, the encoder's named numbers]; written as the site's camera
    move (fold_encoder, the encoder's camera block after the layers)."""
    out_path = Path(out_path)
    _check(out_path.name.endswith(".mlp.json"), f"output must be <name>.mlp.json, got {out_path.name}")
    layers = [(_np(W), _np(b).reshape(-1)) for W, b in layers]
    _check(len(layers) >= 1, "no layers")
    cam_out = 0
    if encoder is not None:
        sub, div, layers = fold_encoder(sub, div, layers, encoder)
        cam_out = encoder.latent_dim
    n_in = layers[0][0].shape[1] - cam_out if layers[0][0].ndim == 2 else IN_DIM
    _check(n_in == IN_DIM, f"input must be {IN_DIM}, got {n_in} (the ray eye's 73-input vision moves were removed, LB-330)")
    sub = np.zeros(n_in) if sub is None else _np(sub).reshape(-1)
    div = np.ones(n_in) if div is None else _np(div).reshape(-1)
    acts = _acts(activation, len(layers))
    _check(sub.shape == (n_in,) and div.shape == (n_in,), f"normaliser must be {n_in} wide: {sub.shape} {div.shape}")
    width = n_in + cam_out
    for k, (W, b) in enumerate(layers):
        _check(W.ndim == 2 and W.shape[1] == width, f"layer {k}: W {W.shape}, expected [out, {width}]")
        _check(b.shape == (W.shape[0],), f"layer {k}: bias {b.shape} for W {W.shape}")
        width = W.shape[0]
    _check(width == OUT_DIM, f"output width {width} != {OUT_DIM}")
    chunks = [sub.astype("<f4"), div.astype("<f4")]
    _check(all(np.isfinite(c).all() for c in chunks), "non-finite normaliser")
    _check(bool((chunks[1] != 0).all()), "a zero in div")
    offset = 2 * n_in
    spec_layers = []
    for (W, b), (kind, alpha) in zip(layers, acts):
        W32, b32 = np.ascontiguousarray(W.astype("<f4")), b.astype("<f4")
        _check(np.isfinite(W32).all() and np.isfinite(b32).all(), f"non-finite weights in layer {len(spec_layers)}")
        spec_layers.append({"in": W.shape[1], "out": W.shape[0], "weight_offset": offset, "bias_offset": offset + W32.size,
                            "weight_layout": "out_major", "activation": kind, "elu_alpha": alpha})
        offset += W32.size + b32.size
        chunks += [W32.reshape(-1), b32]
    camera = None
    if encoder is not None:
        camera, parts = encoder.camera_block(offset)
        chunks += parts
        offset += sum(x.size for x in parts)
    blob = np.concatenate(chunks).astype("<f4")
    assert blob.size == offset
    stem = out_path.name[: -len(".mlp.json")]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (out_path.parent / f"{stem}.mlp.bin").write_bytes(blob.tobytes())
    spec = {"format": "duck-mlp/1", "source_onnx_sha256": source_sha256, "input_dim": n_in, "output_dim": OUT_DIM,
            "norm": {"sub_offset": 0, "div_offset": n_in, "len": n_in}, "layers": spec_layers,
            **({"camera": camera} if camera is not None else {})}
    if meta or encoder is not None:
        spec["meta"] = dict(meta or {}, **({"encoder": {"name": encoder.name, "id": encoder.id, "names": encoder.names}}
                                           if encoder is not None else {}))
    out_path.write_text(json.dumps(spec, indent=1) + "\n")
    return spec


def camera_forward(spec: dict, blob: np.ndarray, pictures: np.ndarray) -> np.ndarray:
    """A camera move's CNN read from its own blob (schema move-camera.ts duck-cam/1; an implementation independent of
    duck_rl/encoder.py): pictures [B, 4, 16, 32] -> features [B, camera.out] float32 (float64 sums, float32 stores)."""
    c = spec["camera"]
    x = np.asarray(pictures, np.float64).reshape(-1, 4, 16, 32)
    elu = lambda v: np.where(v > 0, v, np.expm1(np.minimum(v, 0.0)))  # noqa: E731
    for v in c["convs"]:
        k, st, pd = v["k"], v["stride"], v["pad"]
        W = blob[v["weight_offset"]:v["weight_offset"] + v["out"] * v["in"] * k * k].reshape(v["out"], v["in"], k, k).astype(np.float64)
        b = blob[v["bias_offset"]:v["bias_offset"] + v["out"]].astype(np.float64)
        xp = np.pad(x, ((0, 0), (0, 0), (pd, pd), (pd, pd)))
        oh, ow = (xp.shape[2] - k) // st + 1, (xp.shape[3] - k) // st + 1
        y = np.zeros((x.shape[0], v["out"], oh, ow))
        for ky in range(k):
            for kx in range(k):
                y += np.einsum("oi,biyx->boyx", W[:, :, ky, kx], xp[:, :, ky:ky + st * (oh - 1) + 1:st, kx:kx + st * (ow - 1) + 1:st])
        y = (y + b[None, :, None, None]).astype(np.float32).astype(np.float64)
        x = (elu(y) if v["activation"] == "elu" else y).astype(np.float32).astype(np.float64)
    d = c["dense"]
    W = blob[d["weight_offset"]:d["weight_offset"] + d["out"] * d["in"]].reshape(d["out"], d["in"]).astype(np.float64)
    f = (x.reshape(x.shape[0], -1) @ W.T + blob[d["bias_offset"]:d["bias_offset"] + d["out"]]).astype(np.float32).astype(np.float64)
    return (elu(f) if d["activation"] == "elu" else f).astype(np.float32)


def mlp_forward(json_path, x: np.ndarray, pictures: np.ndarray | None = None) -> np.ndarray:
    """float32 numpy forward of a written duck-mlp/1 (as the ONNX graph: float32 between ops). A camera move also takes
    its pictures [B, 4, 16, 32] (camera_forward; its features join layer 0 after the normalised inputs)."""
    json_path = Path(json_path)
    spec = json.loads(json_path.read_text())
    blob = np.fromfile(json_path.with_name(json_path.name[: -len(".json")] + ".bin"), dtype="<f4")
    n = spec["norm"]
    sub = blob[n["sub_offset"]:n["sub_offset"] + n["len"]]
    div = blob[n["div_offset"]:n["div_offset"] + n["len"]]
    h = ((np.asarray(x, np.float32) - sub) / div).astype(np.float32)
    if "camera" in spec:
        _check(pictures is not None, f"{json_path}: a camera move needs its pictures")
        h = np.concatenate([h.reshape(-1, n["len"]), camera_forward(spec, blob, pictures)], 1)
    for L in spec["layers"]:
        W = blob[L["weight_offset"]:L["weight_offset"] + L["in"] * L["out"]].reshape(L["out"], L["in"])
        b = blob[L["bias_offset"]:L["bias_offset"] + L["out"]]
        h = (h @ W.T + b).astype(np.float32)
        if L["activation"] == "elu":
            a = np.float32(L["elu_alpha"] if L["elu_alpha"] is not None else 1.0)
            h = np.where(h > 0, h, a * np.expm1(h)).astype(np.float32)
    return h


# ------------------------------------------------------------------------------------------------ loading: torch

def _linear_chain(sd: dict, prefix: str | None = None) -> list[tuple]:
    """Ordered (W, b) of every 2-D '<prefix><k>.weight' (+ '.bias') in a state dict."""
    found = {}
    for key, v in sd.items():
        m = re.fullmatch(r"(.*?)(\d+)\.weight", key)
        if m and hasattr(v, "dim") and v.dim() == 2 and (prefix is None or m.group(1) == prefix):
            found.setdefault(m.group(1), []).append((int(m.group(2)), key))
    _check(found, "no '<prefix><k>.weight' 2-D tensors in the state dict")
    _check(len(found) == 1, f"several Linear chains in the state dict: prefixes {sorted(found)}; pass one")
    (pre, items), = found.items()
    layers = []
    for _, key in sorted(items):
        bkey = key[: -len("weight")] + "bias"
        W = sd[key]
        layers.append((W, sd[bkey] if bkey in sd else np.zeros(W.shape[0])))
    return layers


def _find(sd: dict, name: str):
    hits = [k for k in sd if k == name or k.endswith("." + name)]
    _check(len(hits) <= 1, f"ambiguous {name}: {hits}")
    return sd[hits[0]] if hits else None


def _from_state_dict(sd: dict, activation, eps: float, label: str) -> tuple:
    sub, div = _find(sd, "sub"), _find(sd, "div")
    kind = "sub/div"
    if sub is None and div is None and _find(sd, "obs_normalizer._mean") is not None:
        sub, div = _find(sd, "obs_normalizer._mean"), _np(_find(sd, "obs_normalizer._std")) + eps
        kind = f"obs_normalizer (_std + {eps})"
    elif sub is None and div is None:
        kind = "none (sub 0, div 1)"
    prefix = None
    if any(k.startswith("mlp.") for k in sd) and any(k.startswith("critic") for k in sd):
        prefix = "mlp."
    layers = _linear_chain({k: v for k, v in sd.items() if not k.startswith(("critic", "distribution"))}, prefix)
    return sub, div, layers, _acts(activation, len(layers)), f"{label}; normaliser {kind}"


def _module_acts(mod) -> list | None:
    import torch.nn as nn
    seq = mod if isinstance(mod, nn.Sequential) else next((m for m in mod.modules() if isinstance(m, nn.Sequential)), None)
    if seq is None:
        return None
    acts = []
    for m in seq:
        if isinstance(m, nn.Linear):
            acts.append(("none", None))
        elif isinstance(m, nn.ELU):
            _check(acts, "ELU before the first Linear")
            acts[-1] = ("elu", float(m.alpha))
        elif isinstance(m, (nn.Identity, nn.Dropout)):
            pass
        else:
            raise ConvertError(f"module {type(m).__name__}: only Linear / ELU / Identity are supported")
    return acts


def load_torch(path, activation="elu", eps: float | None = None, branch: str | None = None):
    """-> (sub, div, layers, activations, describe). `branch` = "plain" | "actor" to force a checkpoint branch."""
    import torch
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        acts = _module_acts(obj)
        s, d, L, a, desc = _from_state_dict(obj.state_dict(), activation, 1e-2 if eps is None else eps,
                                            f"nn.Module {type(obj).__name__}")
        return s, d, L, (acts if acts is not None else a), desc
    _check(isinstance(obj, dict), f"{path}: a {type(obj).__name__}, expected a dict or an nn.Module")
    if eps is None:  # move-lab checkpoints (learn.py save_ckpt) carry warm_start; learn.warm_start sets nz.eps = 0
        eps = 0.0 if "warm_start" in obj else 1e-2
    if branch is None:
        branch = "plain" if "plain" in obj else "actor" if ("actor" in obj and isinstance(obj["actor"], dict)) else None
    if branch == "plain":
        return _from_state_dict(obj["plain"], "elu", eps, "checkpoint 'plain' (PlainActor)")
    if branch == "actor":
        return _from_state_dict(obj["actor"], "elu", eps, "checkpoint 'actor' (rsl_rl MLPModel)")
    if "layers" in obj:
        layers = [(W, b) for W, b in obj["layers"]]
        return (obj.get("sub"), obj.get("div"), layers, _acts(obj.get("activation", activation), len(layers)),
                "dict {sub, div, layers}")
    return _from_state_dict(obj, activation, eps, "state dict")


# ------------------------------------------------------------------------------------------------ loading: ONNX

_FOLD = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide}


def _fold_onnx(model):
    """Constant -> initializer, Identity bypassed, constant-only nodes evaluated. Returns (inits, nodes)."""
    import onnx
    from onnx import numpy_helper
    g = model.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    rename: dict[str, str] = {}
    nodes = []
    for n in g.node:
        ins = [rename.get(x, x) for x in n.input]
        at = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        if n.op_type == "Constant":
            val = at.get("value")
            _check(val is not None, f"Constant {n.name}: only 'value' supported")
            inits[n.output[0]] = numpy_helper.to_array(val)
            continue
        if n.op_type == "Identity":
            rename[n.output[0]] = ins[0]
            continue
        if ins and all(x in inits or x == "" for x in ins):
            v = [inits[x] for x in ins if x]
            op = n.op_type
            if op in _FOLD:
                r = _FOLD[op](v[0], v[1])
            elif op == "Reshape":
                r = v[0].reshape([d if d != 0 else v[0].shape[i] for i, d in enumerate(v[1].tolist())])
            elif op == "Unsqueeze":
                axes = v[1].tolist() if len(v) > 1 else at["axes"]
                r = np.expand_dims(v[0], tuple(axes))
            elif op == "Squeeze":
                axes = v[1].tolist() if len(v) > 1 else at.get("axes")
                r = np.squeeze(v[0], tuple(axes) if axes is not None else None)
            elif op == "Cast":
                r = v[0].astype(onnx.helper.tensor_dtype_to_np_dtype(at["to"]))
            elif op == "Sqrt":
                r = np.sqrt(v[0])
            elif op == "Transpose":
                r = np.transpose(v[0], at.get("perm"))
            else:
                raise ConvertError(f"constant {op} node {n.name} cannot be folded")
            inits[n.output[0]] = np.asarray(r)
            continue
        n2 = onnx.NodeProto()
        n2.CopyFrom(n)
        del n2.input[:]
        n2.input.extend(ins)
        nodes.append(n2)
    outs = [rename.get(o.name, o.name) for o in g.output]
    return inits, nodes, outs


def load_onnx(path):
    """-> (sub, div, layers, activations, describe, onnx metadata)."""
    import onnx
    model = onnx.load(str(path))
    g = model.graph
    init_names = {i.name for i in g.initializer}
    inputs = [i.name for i in g.input if i.name not in init_names]
    _check(len(inputs) == 1 and len(g.output) == 1, f"want one input and one output, got {inputs} -> {[o.name for o in g.output]}")
    inits, nodes, outs = _fold_onnx(model)
    consumers: dict[str, list] = {}
    for n in nodes:
        for x in n.input:
            consumers.setdefault(x, []).append(n)

    def attrs(n):
        return {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}

    def const_other(n, cur):
        others = [x for x in n.input if x != cur]
        _check(len(others) == 1 and others[0] in inits, f"{n.op_type} {n.name}: needs one constant operand")
        return inits[others[0]].astype(np.float64)

    cur = inputs[0]
    seq = []
    while cur != outs[0]:
        cs = consumers.get(cur, [])
        _check(len(cs) == 1, f"not a chain at tensor {cur!r}: {len(cs)} consumers")
        seq.append((cs[0], cur))
        _check(len(cs[0].output) >= 1, "node without outputs")
        cur = cs[0].output[0]
        _check(len(seq) <= len(nodes), "cycle")
    used = {id(n) for n, _ in seq}
    _check(len(used) == len(nodes), f"graph has nodes off the input->output chain: "
           f"{[n.op_type for n in nodes if id(n) not in used]}")

    sub = div = None
    i = 0
    if i < len(seq) and seq[i][0].op_type == "Sub":
        n, x = seq[i]
        _check(n.input[0] == x, "Sub must be obs - const")
        sub, i = const_other(n, x).reshape(-1), i + 1
    if i < len(seq) and seq[i][0].op_type in ("Div", "Mul"):
        n, x = seq[i]
        c = const_other(n, x).reshape(-1)
        if n.op_type == "Div":
            _check(n.input[0] == x, "Div must be x / const")
            div = c
        else:
            _check(bool((c != 0).all()), "Mul normaliser with a zero factor")
            div = 1.0 / c
        i += 1
    layers, acts = [], []
    while i < len(seq):
        n, x = seq[i]
        if n.op_type == "Gemm":
            a = attrs(n)
            _check(n.input[0] == x and a.get("transA", 0) == 0, f"Gemm {n.name}: must be x @ W (transA 0)")
            _check(n.input[1] in inits, f"Gemm {n.name}: weight not constant")
            W = inits[n.input[1]].astype(np.float64)
            W = W if a.get("transB", 0) else W.T
            b = inits[n.input[2]].astype(np.float64).reshape(-1) if len(n.input) > 2 and n.input[2] else np.zeros(W.shape[0])
            W, b = W * float(a.get("alpha", 1.0)), b * float(a.get("beta", 1.0))
            i += 1
        elif n.op_type == "MatMul":
            _check(n.input[0] == x and n.input[1] in inits, f"MatMul {n.name}: must be x @ const")
            W = inits[n.input[1]].astype(np.float64).T
            b = np.zeros(W.shape[0])
            i += 1
            if i < len(seq) and seq[i][0].op_type == "Add":
                b = const_other(*seq[i]).reshape(-1)
                i += 1
        else:
            raise ConvertError(f"unsupported op {n.op_type} ({n.name}) where a layer (Gemm / MatMul [+Add]) is expected; "
                               "the TS runner supports Sub/Div normaliser, dense layers, ELU")
        _check(b.shape == (W.shape[0],), f"layer {len(layers)}: bias {b.shape} vs W {W.shape}")
        act = ("none", None)
        if i < len(seq) and seq[i][0].op_type == "Elu":
            act = ("elu", float(attrs(seq[i][0]).get("alpha", 1.0)))
            i += 1
        layers.append((W, b))
        acts.append(act)
    _check(layers, "no dense layers")
    meta = {p.key: p.value for p in model.metadata_props}
    desc = f"ONNX ops {[n.op_type for n, _ in seq]}"
    return sub, div, layers, acts, desc, meta


def load_any(path, activation="elu", eps: float | None = None):
    """-> (sub, div, layers[(W[out,in], b[out]) float64], activations[(kind, alpha)]); sub/div None = 0/1."""
    p = Path(path)
    if p.suffix == ".onnx":
        return load_onnx(p)[:4]
    return load_torch(p, activation, eps)[:4]


# ------------------------------------------------------------------------------------------------ validation

def torch_forward(sub, div, layers, acts, x: np.ndarray) -> np.ndarray:
    """Source forward in torch float32, from the source tensors (independent of the blob layout)."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.asarray(x, np.float32))
    h = t if sub is None else t - torch.as_tensor(_np(sub), dtype=torch.float32).reshape(-1)
    h = h if div is None else h / torch.as_tensor(_np(div), dtype=torch.float32).reshape(-1)
    with torch.no_grad():
        for (W, b), (kind, alpha) in zip(layers, acts):
            h = F.linear(h, torch.as_tensor(_np(W), dtype=torch.float32), torch.as_tensor(_np(b), dtype=torch.float32))
            if kind == "elu":
                h = F.elu(h, alpha)
    return h.numpy()


def ort_forward(path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    try:
        return sess.run(None, {name: x})[0]
    except Exception:  # a fixed batch dim of 1
        return np.concatenate([sess.run(None, {name: x[i:i + 1]})[0] for i in range(len(x))])


def ts_check(out_json: Path, x: np.ndarray, ref: np.ndarray, tol: float) -> dict:
    if str(KIT) not in sys.path:
        sys.path.insert(0, str(KIT))
    from duck_rl import bridge  # noqa: PLC0415  (the TS side: the repo's ts/check_mlp.ts, or the CLI's kit check-mlp)
    with tempfile.TemporaryDirectory() as td:
        obs = Path(td) / "obs.npy"
        np.save(obs, x.astype(np.float32))
        r = bridge.run_ts("check-mlp", [Path(out_json).resolve(), obs], capture_output=True, text=True)
    _check(r.returncode == 0, f"check_mlp.ts failed:\n{r.stderr[-2000:]}")
    res = json.loads(r.stdout.strip().splitlines()[-1])
    diffs = {k: float(np.abs(np.asarray(v, np.float32) - ref).max()) for k, v in res["outputs"].items()}
    return {"layers": res["layers"], "diffs": diffs, "ok": diffs["simd"] <= tol}


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert(src, out_json, check: bool = False, activation="elu", eps: float | None = None, ref_onnx=None, n: int = 256,
            branch: str | None = None, meta: dict | None = None, encoder=None) -> dict:
    src, out_json = Path(src), Path(out_json)
    onnx_meta = {}
    if src.suffix == ".onnx":
        sub, div, layers, acts, desc, onnx_meta = load_onnx(src)
    else:
        sub, div, layers, acts, desc = load_torch(src, activation, eps, branch)
    m = {"converter": "spikes/duck-rl/kit/tools/to_mlp.py", "source": src.name, "source_kind": desc}
    if onnx_meta:
        m["onnx_metadata"] = onnx_meta
    m.update(meta or {})
    spec = write_mlp(sub, div, layers, out_json, acts, meta=m, source_sha256=_sha(src), encoder=encoder)
    n_in = spec["input_dim"]
    arch = " -> ".join([str(n_in) + (f" + camera {spec['camera']['out']}" if "camera" in spec else "")]
                       + [f"{L['out']}{'+elu' if L['activation'] == 'elu' else ''}" for L in spec["layers"]])
    print(f"{src} [{desc}] -> {out_json} (+.bin): {arch}")

    rng = np.random.default_rng(0)
    n_src = layers[0][0].shape[1]
    sub_np = np.zeros(n_src) if sub is None else _np(sub).reshape(-1)
    x = (sub_np + 0.5 * rng.normal(size=(n, n_src))).astype(np.float32)  # the source's inputs
    pics = None
    if encoder is not None:  # an encoder move: the source reads [obs61, the encoder's numbers]; the file, obs61 + pictures
        pics = rng.random((n, 4, 16, 32)).astype(np.float32)
        pics[:, 0] = np.where(pics[:, 0] > 0.7, 1.0, pics[:, 0])  # some sky
        x[:, IN_DIM:] = encoder.features(pics)
        x_file = np.ascontiguousarray(x[:, :IN_DIM])
    else:
        x_file = x
    got = mlp_forward(out_json, x_file, pics)
    ref = ort_forward(src, x) if src.suffix == ".onnx" else torch_forward(sub, div, layers, acts, x)
    scale = max(1.0, float(np.abs(ref).max()))
    tol = 1e-4 * scale
    report = {"arch": arch, "max_abs_ref": float(np.abs(ref).max()), "diff_source": float(np.abs(got - ref).max())}
    print(f"  numpy(blob) vs {'onnxruntime' if src.suffix == '.onnx' else 'torch'}(source) on {n} obs: max |diff| "
          f"{report['diff_source']:.2e} (max |out| {report['max_abs_ref']:.3g})")
    ok = report["diff_source"] <= tol
    extra_refs = []
    if ref_onnx is not None:
        extra_refs.append(Path(ref_onnx))
    elif src.suffix == ".pt" and src.with_suffix(".onnx").exists():
        extra_refs.append(src.with_suffix(".onnx"))
    for r in extra_refs:
        d = float(np.abs(got - ort_forward(r, x)).max())
        report[f"diff_{r.name}"] = d
        ok &= d <= tol
        print(f"  numpy(blob) vs onnxruntime({r.name}): max |diff| {d:.2e}")
    if src.suffix == ".pt" and branch is None:
        import torch
        obj = torch.load(str(src), map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "plain" in obj and isinstance(obj.get("actor"), dict):
            s2, d2, l2, a2, _ = load_torch(src, eps=eps, branch="actor")
            d = float(np.abs(got - torch_forward(s2, d2, l2, a2, x)).max())
            report["diff_actor_branch"] = d
            ok &= d <= tol
            print(f"  numpy(blob) vs torch(checkpoint 'actor', eps {"auto" if eps is None else eps}): max |diff| {d:.2e}")
    if check and encoder is not None:
        print("  TS check: not for a camera move (check_mlp.ts reads obs only; tools/camera_check.py runs it in the site sim)")
    elif check:
        ts = ts_check(out_json, x, got, tol)
        report["ts"] = ts["diffs"]
        ok &= ts["ok"]
        print(f"  TS check_mlp.ts (parseMlp {ts['layers']}) vs numpy: " + ", ".join(f"{k} {v:.2e}" for k, v in ts["diffs"].items()))
    report["ok"] = bool(ok)
    print(f"  {'OK' if ok else 'FAIL'} (tolerance {tol:.1e})")
    _check(ok, f"validation failed: {report}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input")
    ap.add_argument("out", help="<name>.mlp.json (the .mlp.bin goes next to it)")
    ap.add_argument("--check", action="store_true", help="also run the TS runners (node spikes/duck-rl/ts/check_mlp.ts)")
    ap.add_argument("--activation", default="elu", choices=["elu", "none"],
                    help="between layers, for state dicts that do not record it (default elu)")
    ap.add_argument("--eps", type=float, default=None,
                    help="obs_normalizer eps for an rsl_rl actor (div = _std + eps); default auto: 0 move-lab ckpt, else 1e-2")
    ap.add_argument("--branch", choices=["plain", "actor"], help="checkpoint branch (default: plain if present)")
    ap.add_argument("--ref", help="an extra ONNX to compare against (default: <ckpt>.onnx next to a .pt, if present)")
    ap.add_argument("--n", type=int, default=256, help="random obs for validation")
    ap.add_argument("--encoder", help="an encoder move (train.py ENCODER, e.g. emb1): write the site's camera move")
    a = ap.parse_args(argv)
    try:
        enc = None
        if a.encoder:
            import importlib.util
            sp = importlib.util.spec_from_file_location("duck_rl_encoder", KIT / "duck_rl" / "encoder.py")
            mod = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(mod)  # type: ignore[union-attr]  (numpy only: no package import-time setup)
            enc = mod.load(a.encoder)
        out = Path(a.out).expanduser()
        convert(Path(a.input).expanduser(), out, a.check, a.activation, a.eps,
                a.ref and Path(a.ref).expanduser(), a.n, a.branch, encoder=enc)
    except ConvertError as e:
        print(f"to_mlp: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
