"""Validation of a duck-sense/1 file: the TS `senseProblem` (packages/schema/src/sense.ts) through
spikes/duck-rl/ts/check_sense.ts when the kit sits in the repo, else `problem()` below, a Python mirror of it (same order of
checks, same caps), plus the loader's value checks (finite weights, std > 0) and MACs against the 512k budget.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

from ..encoder import RETIRED_PICTURES

FORMAT = "duck-sense/1"
PICTURE = "ray-rgbd-palette-4x16x32/2"
PICTURE_SHAPE = (4, 16, 32)
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,23}$")
OUTPUTS_MAX = 16
OPS_MAX = 64  # SENSE_OPS_MAX
LAYERS_MAX = 16  # SENSE_LAYERS_MAX (= MOVE_LAYERS_MAX)
STD_MIN = 1e-6  # SENSE_STD_MIN, compared as float32
ABS_MAX = 1e6  # SENSE_ABS_MAX (= MOVE_ABS_MAX): every .bin value
OP_RAY_MACS = {"depth": 3, "class": 3}  # SENSE_OP_RAY_MACS: an op's charge per ray of its region
CLASS_DECODE_MACS = 512 * 23  # SENSE_CLASS_DECODE_MACS: the class plane's decode, once per strategy with any class_frac op
MACS_PER_TICK = 512_000  # SENSE_MACS_PER_TICK (@duckbench/schema)
BODY_GROUPS = {"gyro": 3, "gravity": 3, "joint_pos": 14, "joint_vel": 14}
BODY_SCALARS = ["t", "tilt", "upright", "fallen", "yaw_rate", "heading", "forward_speed", "side_speed", "speed"]
BODY_INPUTS = [*BODY_GROUPS, *BODY_SCALARS]
CAM_STANDARD = ["cam_free_left", "cam_free_ahead", "cam_free_right", "cam_free_known", "cam_duck_seen", "cam_duck_bearing", "cam_duck_dist"]
CAM_INTERMEDIATE = ["cam_drop_left", "cam_drop_ahead", "cam_drop_right", "cam_drop_known", "cam_obstacle_ahead"]
CAM_INPUTS = CAM_STANDARD + CAM_INTERMEDIATE
MLP_ACTS = ["tanh", "elu", "relu", "none"]
OUT_ACTS = ["linear", "tanh", "sigmoid"]
DEPTH_OPS = ["min", "max", "mean"]
_PAL = json.loads((Path(__file__).resolve().parents[1] / "palette.json").read_text())
CLASSES = [c for c, _ in sorted(_PAL["ids"].items(), key=lambda kv: kv[1]) if c != "self_accent"]
CONVS_MAX, CHANNELS_MAX, KERNEL_MAX, STRIDE_MAX, CAM_OUT_MAX = 4, 64, 7, 4, 256
TOP_KEYS = ["format", "name", "picture", "inputs", "conv", "ops", "norm", "mlp", "outputs", "gate", "definition", "eval"]
OFFSET_MAX = 2 ** 31 - 1


def _is_int(v, lo, hi) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _extra(o: dict, keys) -> str | None:
    return next((k for k in o if k not in keys), None)


def body_width(n: str) -> int:
    return BODY_GROUPS.get(n, 1)


def current_count(j: dict) -> int:
    return sum(body_width(n) for n in j["inputs"]["body"]) + len(j["inputs"]["cam"]) + len(j.get("ops") or [])


def vector_count(j: dict) -> int:
    c = current_count(j)
    pc = j["conv"]["dense"]["out"] if j["inputs"].get("prev_conv") and "conv" in j else 0
    return c + (c if j["inputs"]["prev"] else 0) + pc + (len(j["outputs"]) if j["inputs"]["prev_out"] else 0)


def conv_macs(j: dict) -> int:
    c = j.get("conv")
    if c is None:
        return 0
    _, h, w = PICTURE_SHAPE
    macs = c["dense"]["in"] * c["dense"]["out"]
    for v in c["convs"]:
        h = (h + 2 * v["pad"] - v["k"]) // v["stride"] + 1
        w = (w + 2 * v["pad"] - v["k"]) // v["stride"] + 1
        macs += v["out"] * h * w * v["in"] * v["k"] * v["k"]
    return macs


def op_macs(o: dict) -> int:
    rays = (o["rows"][1] - o["rows"][0] + 1) * (o["cols"][1] - o["cols"][0] + 1)
    return rays * OP_RAY_MACS["class" if o["op"] == "class_frac" else "depth"]


def class_macs(j: dict) -> int:
    return CLASS_DECODE_MACS if any(o["op"] == "class_frac" for o in j.get("ops") or []) else 0


def own_macs(j: dict) -> int:
    return sum(op_macs(o) for o in j.get("ops") or []) + sum(l["in"] * l["out"] for l in j["mlp"])


def macs(j: dict) -> int:
    return conv_macs(j) + own_macs(j) + class_macs(j)


FREE_FIELDS = ("definition", "eval")  # SENSE_FREE_FIELDS: outside the id


def _js_number(x) -> str:
    """A number as JavaScript's JSON.stringify writes it (shortest round-trip digits, ECMAScript Number::toString)."""
    if isinstance(x, bool):
        raise TypeError("bool is not a number")
    if isinstance(x, int):
        x = float(x) if abs(x) >= 2 ** 53 else x
        if isinstance(x, int):
            return str(x)
    if not math.isfinite(x):
        raise ValueError("canonical JSON: non-finite number")
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    r = repr(abs(x))  # shortest round-trip digits
    mant, _, exp = r.partition("e")
    e = int(exp) if exp else 0
    ip, _, fp = mant.partition(".")
    digits = (ip + fp).lstrip("0")
    lead = len(ip + fp) - len((ip + fp).lstrip("0"))
    n = len(ip) + e - lead  # the decimal point sits after n digits
    digits = digits.rstrip("0")
    k = len(digits)
    if k <= n <= 21:
        out = digits + "0" * (n - k)
    elif 0 < n <= 21:
        out = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        out = "0." + "0" * (-n) + digits
    else:
        out = digits[0] + ("." + digits[1:] if k > 1 else "") + "e" + ("+" if n - 1 >= 0 else "-") + str(abs(n - 1))
    return sign + out


def canonical_json(v) -> str:
    """schema canonical.ts canonicalJson: keys sorted, no whitespace, numbers as JSON.stringify."""
    if v is None or isinstance(v, bool):
        return json.dumps(v)
    if isinstance(v, (int, float)):
        return _js_number(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ",".join(canonical_json(a) for a in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(f"{json.dumps(k, ensure_ascii=False)}:{canonical_json(v[k])}" for k in sorted(v)) + "}"
    raise TypeError(f"canonical JSON: unsupported {type(v).__name__}")


def id_json(spec) -> str:
    """schema senseIdJson: the behavioural fields (all but FREE_FIELDS) as canonical JSON."""
    if isinstance(spec, dict):
        spec = {k: v for k, v in spec.items() if k not in FREE_FIELDS}
    return canonical_json(spec)


def sense_id(spec, bin_bytes: bytes) -> str:
    """schema senseId: "sn_" + sha256(id_json(spec) as UTF-8 + bin bytes)[:12]. `spec`: the parsed .sense.json."""
    return "sn_" + hashlib.sha256(id_json(spec).encode("utf-8") + bin_bytes).hexdigest()[:12]


def _camera_problem(c: dict, picture) -> str | None:
    if isinstance(picture, str) and picture in RETIRED_PICTURES:
        return f'picture "{picture}" is retired: {RETIRED_PICTURES[picture]}'
    convs = c.get("convs")
    if not isinstance(convs, list) or not 1 <= len(convs) <= CONVS_MAX:
        return f"convs must be an array of 1-{CONVS_MAX} convolutions"
    ch, h, w = PICTURE_SHAPE
    for k, v in enumerate(convs):
        at = f"convs[{k}]"
        if not isinstance(v, dict):
            return f"{at} must be an object"
        if not _is_int(v.get("in"), 1, CHANNELS_MAX) or not _is_int(v.get("out"), 1, CHANNELS_MAX):
            return f"{at}.in and .out must be integers 1..{CHANNELS_MAX}"
        if not _is_int(v.get("k"), 1, KERNEL_MAX):
            return f"{at}.k must be an integer 1..{KERNEL_MAX}"
        if not _is_int(v.get("stride"), 1, STRIDE_MAX):
            return f"{at}.stride must be an integer 1..{STRIDE_MAX}"
        if not _is_int(v.get("pad"), 0, v["k"] - 1):
            return f"{at}.pad must be an integer 0..k-1"
        for o in ("weight_offset", "bias_offset"):
            if not _is_int(v.get(o), 0, OFFSET_MAX):
                return f"{at}.{o} must be a non-negative integer"
        if v.get("activation") not in ("elu", "none"):
            return f'{at}.activation must be "elu" or "none"'
        e = _extra(v, ["in", "out", "k", "stride", "pad", "activation", "weight_offset", "bias_offset"])
        if e is not None:
            return f'{at} has an unknown key "{e}"'
        if v["in"] != ch:
            return f"{at}.in is {v['in']}; its input has {ch} channels"
        kk, s, p = v["k"], v["stride"], v["pad"]
        if h + 2 * p < kk or w + 2 * p < kk:
            return f"{at}: the {kk} x {kk} kernel does not fit its {h} x {w} input"
        h, w, ch = (h + 2 * p - kk) // s + 1, (w + 2 * p - kk) // s + 1, v["out"]
    d = c.get("dense")
    if not isinstance(d, dict):
        return "dense must be an object"
    if not _is_int(d.get("in"), 1, 2 ** 20) or not _is_int(d.get("out"), 1, CAM_OUT_MAX):
        return f"dense.in must be a positive integer and dense.out an integer 1..{CAM_OUT_MAX}"
    for o in ("weight_offset", "bias_offset"):
        if not _is_int(d.get(o), 0, OFFSET_MAX):
            return f"dense.{o} must be a non-negative integer"
    if d.get("activation") not in ("elu", "none"):
        return 'dense.activation must be "elu" or "none"'
    e = _extra(d, ["in", "out", "activation", "weight_offset", "bias_offset"])
    if e is not None:
        return f'dense has an unknown key "{e}"'
    if d["in"] != ch * h * w:
        return f"dense.in is {d['in']}; the last convolution gives {ch} x {h} x {w} = {ch * h * w}"
    return None


def _ranges(j: dict) -> list[tuple[int, int, str]]:
    out = []
    c = j.get("conv")
    if c is not None:
        for k, v in enumerate(c["convs"]):
            out += [(v["weight_offset"], v["out"] * v["in"] * v["k"] ** 2, f"conv.convs[{k}].weight"), (v["bias_offset"], v["out"], f"conv.convs[{k}].bias")]
        d = c["dense"]
        out += [(d["weight_offset"], d["in"] * d["out"], "conv.dense.weight"), (d["bias_offset"], d["out"], "conv.dense.bias")]
    nv = vector_count(j)
    out += [(j["norm"]["mean_offset"], nv, "norm.mean"), (j["norm"]["std_offset"], nv, "norm.std")]
    for k, l in enumerate(j["mlp"]):
        out += [(l["weight_offset"], l["in"] * l["out"], f"mlp[{k}].weight"), (l["bias_offset"], l["out"], f"mlp[{k}].bias")]
    return out


def problem(j, bin_bytes: int) -> str | None:
    """Python mirror of senseProblem: why `j` is not a valid duck-sense/1 spec with a .bin of `bin_bytes`, or None."""
    if not isinstance(j, dict):
        return "a sense must be a JSON object"
    if j.get("format") != FORMAT:
        return f'format must be "{FORMAT}", got {json.dumps(j.get("format"))}'
    e = _extra(j, TOP_KEYS)
    if e is not None:
        return f'unknown key "{e}"'
    if not isinstance(j.get("name"), str) or not NAME_RE.match(j["name"]):
        return f"name must match {NAME_RE.pattern}, got {json.dumps(j.get('name'))}"
    inp = j.get("inputs")
    if not isinstance(inp, dict):
        return "inputs must be an object { body, cam, prev, prev_conv?, prev_out }"
    e = _extra(inp, ["body", "cam", "prev", "prev_conv", "prev_out"])
    if e is not None:
        return f'inputs has an unknown key "{e}"'
    if not isinstance(inp.get("body"), list) or not all(isinstance(s, str) for s in inp["body"]):
        return "inputs.body must be an array of names"
    for n in inp["body"]:
        if n not in BODY_INPUTS:
            return f'inputs.body: "{n}" is not a body input; allowed: {", ".join(BODY_INPUTS)}'
    if len(set(inp["body"])) != len(inp["body"]):
        return "inputs.body names a signal twice"
    if not isinstance(inp.get("cam"), list) or not all(isinstance(s, str) for s in inp["cam"]):
        return "inputs.cam must be an array of names"
    for n in inp["cam"]:
        if n not in CAM_INPUTS:
            return f'inputs.cam: "{n}" is not a camera input; allowed: {", ".join(CAM_INPUTS)}'
    if len(set(inp["cam"])) != len(inp["cam"]):
        return "inputs.cam names a signal twice"
    if not isinstance(inp.get("prev"), bool) or not isinstance(inp.get("prev_out"), bool):
        return "inputs.prev and inputs.prev_out must be true or false"
    if "prev_conv" in inp and not isinstance(inp["prev_conv"], bool):
        return "inputs.prev_conv must be true or false"
    if inp.get("prev_conv") and "conv" not in j:
        return "inputs.prev_conv needs a conv block"
    sees = "conv" in j or "ops" in j
    if "picture" not in j:
        if sees:
            return f'picture is required with conv or ops: "{PICTURE}"'
    else:
        if not sees:
            return "picture is given but neither conv nor ops reads it"
        if isinstance(j["picture"], str) and j["picture"] in RETIRED_PICTURES:
            return f'picture "{j["picture"]}" is retired: {RETIRED_PICTURES[j["picture"]]}'
        if j["picture"] != PICTURE:
            return f'picture must be "{PICTURE}", got {json.dumps(j["picture"])}'
    if "conv" in j:
        if not isinstance(j["conv"], dict):
            return "conv must be an object { convs, dense }"
        e = _extra(j["conv"], ["convs", "dense"])
        if e is not None:
            return f'conv has an unknown key "{e}"'
        p = _camera_problem(j["conv"], j.get("picture"))
        if p is not None:
            return f"conv: {p}"
    if "ops" in j:
        if not isinstance(j["ops"], list):
            return "ops must be an array"
        if len(j["ops"]) > OPS_MAX:
            return f"ops has {len(j['ops'])} ops; a sense takes at most {OPS_MAX}"
        for k, o in enumerate(j["ops"]):
            at = f"ops[{k}]"
            if not isinstance(o, dict):
                return f"{at} must be an object"
            if o.get("op") in DEPTH_OPS:
                e = _extra(o, ["op", "channel", "rows", "cols"])
                if e is not None:
                    return f'{at} has an unknown key "{e}"'
                if o.get("channel") != "depth":
                    return f'{at}.channel must be "depth"'
            elif o.get("op") == "class_frac":
                e = _extra(o, ["op", "class", "rows", "cols"])
                if e is not None:
                    return f'{at} has an unknown key "{e}"'
                if o.get("class") == "self_accent":
                    return f'{at}.class "self_accent" has the same colour as "self"; use "self"'
                if o.get("class") not in CLASSES:
                    return f"{at}.class must be one of {', '.join(CLASSES)}, got {json.dumps(o.get('class'))}"
            else:
                return f"{at}.op must be one of {', '.join(DEPTH_OPS + ['class_frac'])}, got {json.dumps(o.get('op'))}"
            for key, size in (("rows", PICTURE_SHAPE[1]), ("cols", PICTURE_SHAPE[2])):
                r = o.get(key)
                if not (isinstance(r, list) and len(r) == 2 and _is_int(r[0], 0, size - 1) and _is_int(r[1], 0, size - 1) and r[0] <= r[1]):
                    return f"{at}.{key} must be [first, last], integers 0..{size - 1}, first <= last (inclusive)"
    outs = j.get("outputs")
    if not isinstance(outs, list) or not 1 <= len(outs) <= OUTPUTS_MAX:
        return f"outputs must be an array of 1-{OUTPUTS_MAX} outputs"
    names = set()
    for k, o in enumerate(outs):
        at = f"outputs[{k}]"
        if not isinstance(o, dict):
            return f"{at} must be an object"
        e = _extra(o, ["name", "unit", "min", "max", "none", "activation"])
        if e is not None:
            return f'{at} has an unknown key "{e}"'
        if not isinstance(o.get("name"), str) or not NAME_RE.match(o["name"]):
            return f"{at}.name must match {NAME_RE.pattern}, got {json.dumps(o.get('name'))}"
        if o["name"] in names:
            return f'{at}.name "{o["name"]}" is used twice'
        names.add(o["name"])
        if not isinstance(o.get("unit"), str):
            return f'{at}.unit must be a string ("" for none)'
        if not (_num(o.get("min")) and _num(o.get("max")) and math.isfinite(o["min"]) and math.isfinite(o["max"]) and o["min"] <= o["max"]):
            return f"{at}.min and .max must be finite numbers, min <= max"
        nn = o.get("none")
        if not ((_num(nn) and math.isfinite(nn)) or nn in ("Infinity", "-Infinity")):
            return f'{at}.none must be a finite number, "Infinity" or "-Infinity"'
        if o.get("activation") not in OUT_ACTS:
            return f"{at}.activation must be one of {', '.join(OUT_ACTS)}"
    if "gate" in j and not _is_int(j["gate"], 0, len(outs) - 1):
        return f"gate must be an output index 0..{len(outs) - 1}"
    if "definition" in j:
        d = j["definition"]
        if not isinstance(d, dict) or not isinstance(d.get("label"), str) or not isinstance(d.get("collect"), str):
            return "definition must be { label: string, collect: string }"
        e = _extra(d, ["label", "collect"])
        if e is not None:
            return f'definition has an unknown key "{e}"'
    nm = j.get("norm")
    if not isinstance(nm, dict):
        return "norm must be an object { mean_offset, std_offset }"
    e = _extra(nm, ["mean_offset", "std_offset"])
    if e is not None:
        return f'norm has an unknown key "{e}"'
    if not _is_int(nm.get("mean_offset"), 0, OFFSET_MAX) or not _is_int(nm.get("std_offset"), 0, OFFSET_MAX):
        return "norm.mean_offset and norm.std_offset must be non-negative integers"
    n_feat = (j["conv"]["dense"]["out"] if "conv" in j else 0) + vector_count(j)
    if n_feat < 1:
        return "the sense has no inputs (no conv, body, cam, ops or previous outputs)"
    if not isinstance(j.get("mlp"), list) or len(j["mlp"]) < 1:
        return "mlp must be an array of at least one layer"
    if len(j["mlp"]) > LAYERS_MAX:
        return f"mlp has {len(j['mlp'])} layers; a sense takes at most {LAYERS_MAX}"
    width = n_feat
    for k, l in enumerate(j["mlp"]):
        at = f"mlp[{k}]"
        if not isinstance(l, dict):
            return f"{at} must be an object"
        e = _extra(l, ["in", "out", "weight_offset", "bias_offset", "activation"])
        if e is not None:
            return f'{at} has an unknown key "{e}"'
        if not _is_int(l.get("in"), 1, 2 ** 20) or not _is_int(l.get("out"), 1, 2 ** 20):
            return f"{at}.in and .out must be positive integers"
        if not _is_int(l.get("weight_offset"), 0, OFFSET_MAX) or not _is_int(l.get("bias_offset"), 0, OFFSET_MAX):
            return f"{at}.weight_offset and .bias_offset must be non-negative integers"
        if l.get("activation") not in MLP_ACTS:
            return f"{at}.activation must be one of {', '.join(MLP_ACTS)}"
        if l["in"] != width:
            return (f"mlp[0].in is {l['in']}; the features are {n_feat}" if k == 0 else f"{at}.in is {l['in']}; mlp[{k - 1}].out is {width}")
        width = l["out"]
    if width != len(outs):
        return f"the last layer gives {width} values; there are {len(outs)} outputs"
    if bin_bytes < 0 or bin_bytes % 4:
        return f"the .bin must be whole float32s, got {bin_bytes} bytes"
    floats = bin_bytes // 4
    for off, n, label in _ranges(j):
        if off + n > floats:
            return f"{label} [{off}, {off + n}) is outside the .bin ({floats} floats)"
    m = macs(j)
    if m > MACS_PER_TICK:
        return f"{m} multiply-adds per tick; the budget is {MACS_PER_TICK} for all of a strategy's senses"
    return None


def value_problem(j: dict, w: np.ndarray) -> str | None:
    """schema senseWeightsProblem (the sim's loader, the CLI, the server): every value finite and |v| <= ABS_MAX, every std >= STD_MIN."""
    bad = np.flatnonzero(~(np.abs(w.astype(np.float64)) <= ABS_MAX))
    if bad.size:
        return f"the .bin's value {int(bad[0])} is {float(w[bad[0]])}: every value must be finite and at most {int(ABS_MAX)} in magnitude"
    nv = vector_count(j)
    sd = w[j["norm"]["std_offset"]:j["norm"]["std_offset"] + nv]
    if not np.all(sd >= np.float32(STD_MIN)):
        return f"norm.std has a value below {STD_MIN}"
    return None


def paths(sense: str | Path) -> tuple[Path, Path]:
    """<name> | <name>.sense.json -> (json, bin)."""
    p = Path(sense).expanduser()
    if not p.name.endswith(".sense.json"):
        p = p.with_name(p.name + ".sense.json")
    return p, p.with_name(p.name[: -len(".json")] + ".bin")


def ts_check(jp: Path, bp: Path) -> dict | None:
    """The repo's TS check (senseProblem + the sim's loader), or None outside the repo / without node."""
    from .. import bridge
    if bridge.REPO is None:
        return None
    script = Path(bridge.REPO) / "spikes/duck-rl/ts/check_sense.ts"
    if not script.exists():
        return None
    try:
        r = subprocess.run(["node", str(script), str(jp)], capture_output=True, text=True, timeout=60, cwd=bridge.REPO)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode not in (0, 1) or not r.stdout.strip():
        return None
    return json.loads(r.stdout.strip().splitlines()[-1])


def check(sense: str | Path, ts: bool = True) -> dict:
    """{ok, problem, macs, budget, id, via} for a sense file."""
    jp, bp = paths(sense)
    jb, bb = jp.read_bytes(), bp.read_bytes() if bp.exists() else b""
    j = json.loads(jb)
    res = {"sense": str(jp), "id": sense_id(j, bb), "budget": MACS_PER_TICK}
    t = ts_check(jp, bp) if ts else None
    if t is not None:
        res.update(via="ts (packages/schema senseProblem + sim Sense)", problem=t.get("problem"), macs=t.get("macs"))
        if t.get("id") and t["id"] != res["id"]:
            res["problem"] = res["problem"] or f"id mismatch: ts {t['id']} vs python {res['id']}"
    else:
        p = problem(j, len(bb))
        if p is None:
            p = value_problem(j, np.frombuffer(bb, "<f4"))
        res.update(via="python mirror of senseProblem", problem=p, macs=macs(j) if p is None or "multiply-adds" in p else None)
    res["ok"] = res["problem"] is None
    return res
