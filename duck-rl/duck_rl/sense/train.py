"""Train a player's sense (duck-sense/1, docs/senses-design.md §5a) on `duck-sense-collect/1` data with a label file.

Templates (TEMPLATES):
  mlp-cam    the default: a small MLP (2 x 32 tanh) over cam_* + body signals + the previous tick; no picture; seconds
  emb1-head  the kit's frozen encoder emb1 (duck_rl/encoders/emb1: 4 convs + dense 64, 222k MACs) copied into the sense's
             `conv` block, its 64 numbers + the same vector into a small MLP; only the MLP trains
  conv       a small conv stack (3 convs + dense 32, ~130k MACs) trained from the picture with the MLP

The label file is Python: `label(row) -> float | bool | None | dict[str, float | None]` (data.Row: row.world, row.at(dt),
row.prev, row.seed, row.t, row.challenge, row.duck); a row whose label is None is skipped (a dict: per output). Only the
strategy's own duck's rows (row.duck == 0) are labelled unless ducks="all" (--all-ducks). Optional names in it:
  OUTPUTS = {"near": {"activation": "sigmoid"|"linear"|"tanh", "unit": "", "min": .., "max": .., "none": ..}}
  INPUTS  = {"body": [...], "cam": [...], "prev": True}      (else the template's defaults over what the data has)
An output whose labels are all 0 / 1 (or booleans) is sigmoid (BCE), else linear (Huber on the standardised target).

Splits by seed (a (directory, seed) pair is one unit): test 20 %, validation 15 % of the rest (early stop), train the rest.
The exported file is evaluated by the reference evaluator (duck_rl/sense/ref.py, the twin of sim sense-eval.ts) on
held-out sequences and must equal the torch forward pass within 1e-5 (eval.parity).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from . import check as C
from .data import Data
from .ref import Sense as RefSense

TEMPLATES = {
    "mlp-cam": {"conv": None, "hidden": [32, 32], "act": "tanh", "epochs": 300, "patience": 20, "batch": 1024, "lr": 3e-3},
    "emb1-head": {"conv": "emb1", "hidden": [32, 32], "act": "tanh", "epochs": 300, "patience": 20, "batch": 1024, "lr": 3e-3},
    "conv": {"conv": "train", "hidden": [32], "act": "tanh", "epochs": 80, "patience": 8, "batch": 256, "lr": 2e-3,
             # (in, out, k, stride, pad): 16x32x8 -> 8x16x16 -> 4x8x16, dense 512 -> 32 (ELU)
             "convs": [(4, 8, 1, 1, 0), (8, 16, 2, 2, 0), (16, 16, 2, 2, 0)], "dense": 32},
}
DEFAULT_BODY = ["gyro", "gravity", "tilt", "upright", "yaw_rate", "forward_speed", "side_speed", "speed"]
PARITY_TOL = 1e-5


# ---------------------------------------------------------------------------------------------------------- the label


def load_label(src: str, where: str = "<label>") -> dict:
    ns: dict = {"__name__": "duck_sense_label", "__file__": where, "math": math, "np": np}
    exec(compile(src, where, "exec"), ns)
    if not callable(ns.get("label")):
        raise SystemExit(f"{where}: defines no label(row) function")
    return ns


def run_label(ns: dict, d: Data, outputs: dict | None, ducks: str = "own") -> tuple[list[str], np.ndarray]:
    """Labels [n, n_out] (NaN = skipped) and the output names. ducks "own": only duck 0's rows (the strategy's), "all"."""
    if ducks not in ("own", "all"):
        raise SystemExit(f"ducks must be own or all, not {ducks!r}")
    fn = ns["label"]
    vals = [fn(d.row(i)) if ducks == "all" or d.duck[i] == 0 else None for i in range(d.n)]
    is_dict = any(isinstance(v, dict) for v in vals)
    if is_dict:
        names = list(outputs) if outputs else sorted({k for v in vals if isinstance(v, dict) for k in v})
    else:
        names = list(outputs)[:1] if outputs else ["value"]
    Y = np.full((d.n, len(names)), np.nan)
    for i, v in enumerate(vals):
        if v is None:
            continue
        if is_dict:
            for k, nm in enumerate(names):
                x = v.get(nm)
                if x is not None:
                    Y[i, k] = float(x)
        else:
            Y[i, 0] = float(v)
    for k, nm in enumerate(names):
        if not C.NAME_RE.match(nm):
            raise SystemExit(f"output name {nm!r} must match {C.NAME_RE.pattern}")
        if np.isinf(Y[:, k]).any():
            raise SystemExit(f"output {nm}: a label is infinite")
    return names, Y


def output_specs(names: list[str], Y: np.ndarray, declared: dict | None) -> list[dict]:
    outs = []
    for k, nm in enumerate(names):
        y = Y[:, k][~np.isnan(Y[:, k])]
        if not len(y):
            raise SystemExit(f"output {nm}: every label is None")
        dec = dict((declared or {}).get(nm) or {})
        act = dec.get("activation") or ("sigmoid" if np.isin(y, (0.0, 1.0)).all() else "linear")
        if act == "sigmoid":
            lo, hi = 0.0, 1.0
        elif act == "tanh":
            lo, hi = -1.0, 1.0
        else:
            span = float(y.max() - y.min()) or 1.0
            lo, hi = float(y.min()) - 0.1 * span, float(y.max()) + 0.1 * span
        o = {"name": nm, "unit": dec.get("unit", ""), "min": float(dec.get("min", lo)), "max": float(dec.get("max", hi)),
             "none": dec.get("none", 0.0 if act != "linear" else round(float(np.median(y)), 6)), "activation": act}
        outs.append(o)
    return outs


# ---------------------------------------------------------------------------------------------------------- splits, metrics


def split(d: Data, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit (Challenge id, seed) per row -> train / val / test masks: one seed of one Challenge (its rows from every
    collect directory, e.g. two strategies or a re-collect) sits in one split only."""
    chal = [d.dirs[k].challenge for k in d.dir_of.tolist()]
    u = list(zip(chal, d.seed.tolist()))
    units = sorted(set(u))
    if len(units) < 3:
        raise SystemExit(f"the data has {len(units)} seed(s); a held-out split needs at least 3 (collect more seeds)")
    rng = np.random.default_rng(seed)
    order = [units[i] for i in rng.permutation(len(units))]
    n_test = max(1, round(0.2 * len(units)))
    n_val = max(1, round(0.15 * (len(units) - n_test)))
    test, val = set(order[:n_test]), set(order[n_test:n_test + n_val])
    te = np.array([x in test for x in u])
    va = np.array([x in val for x in u])
    return ~(te | va), va, te


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC of p for labels y >= 0.5 (ties averaged); NaN with one class only."""
    pos = y >= 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(outs: list[dict], Y: np.ndarray, P: np.ndarray, base: np.ndarray) -> dict:
    res = {}
    for k, o in enumerate(outs):
        m = ~np.isnan(Y[:, k])
        y, p = Y[m, k], P[m, k]
        if o["activation"] == "sigmoid":
            res[o["name"]] = {"auc": _r(auc(y, p)), "positive": _r(float((y >= 0.5).mean()) if len(y) else float("nan")), "n": int(m.sum())}
        else:
            res[o["name"]] = {"mae": _r(float(np.abs(y - p).mean()) if len(y) else float("nan")),
                              "mae_constant": _r(float(np.abs(y - base[k]).mean()) if len(y) else float("nan")), "n": int(m.sum())}
    return res


def _r(x: float, nd: int = 4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


# ---------------------------------------------------------------------------------------------------------- the model


def _torch():
    import torch
    torch.set_num_threads(4)
    return torch


def build_model(tpl: dict, n_conv_in: int, n_vec: int, n_out: int):
    torch = _torch()
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = None
            n_feat = n_conv_in
            if tpl["conv"] == "train":
                layers = []
                for cin, cout, k, s, p in tpl["convs"]:
                    layers += [nn.Conv2d(cin, cout, k, s, p), nn.ELU()]
                self.conv = nn.Sequential(*layers)
                h, w, c = 16, 32, 4
                for _, cout, k, s, p in tpl["convs"]:
                    h, w, c = (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1, cout
                self.dense = nn.Linear(c * h * w, tpl["dense"])
                n_feat = tpl["dense"]
            dims = [n_feat + n_vec, *tpl["hidden"], n_out]
            self.mlp = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
            self.act = {"tanh": torch.tanh, "elu": nn.functional.elu, "relu": torch.relu}[tpl["act"]]

        def features(self, cin):
            if self.conv is None:
                return cin
            return nn.functional.elu(self.dense(self.conv(cin).flatten(1)))

        def forward(self, vec, cin=None):
            h = vec if cin is None else torch.cat([self.features(cin), vec], 1)
            for k, l in enumerate(self.mlp):
                h = l(h)
                if k < len(self.mlp) - 1:
                    h = self.act(h)
            return h

    torch.manual_seed(0)
    return Net()


def _outputs_from_z(z: np.ndarray, outs: list[dict], ymu: np.ndarray, ysd: np.ndarray) -> np.ndarray:
    """Pre-activation z (standardised for linear outputs) -> the sense's outputs (activation, clamp)."""
    y = np.empty_like(z, dtype=np.float64)
    for k, o in enumerate(outs):
        a = o["activation"]
        v = z[:, k].astype(np.float64)
        v = 1 / (1 + np.exp(-v)) if a == "sigmoid" else np.tanh(v) if a == "tanh" else v * ysd[k] + ymu[k]
        y[:, k] = np.clip(v, o["min"], o["max"])
    return y


def fit(tpl: dict, V: np.ndarray, CIN, Y: np.ndarray, outs: list[dict], tr: np.ndarray, va: np.ndarray, log=print):
    """Train; returns (model, ymu, ysd, epochs run). V standardised [n, nv] float32; CIN conv input (features or pictures)
    or None; Y labels with NaN; tr / va row index arrays."""
    torch = _torch()
    n_out = len(outs)
    lin = np.array([o["activation"] == "linear" for o in outs])
    ymu = np.array([np.nanmean(Y[tr, k]) if lin[k] else 0.0 for k in range(n_out)])
    ysd = np.array([max(np.nanstd(Y[tr, k]), 1e-6) if lin[k] else 1.0 for k in range(n_out)])
    Yt = (Y - ymu) / ysd
    n_cin = 0 if CIN is None or tpl["conv"] == "train" else CIN.shape[1]
    model = build_model(tpl, n_cin, V.shape[1], n_out)
    opt = torch.optim.Adam(model.parameters(), lr=tpl["lr"], weight_decay=1e-5)
    Vt = torch.from_numpy(np.ascontiguousarray(V, np.float32))
    Ct = None if CIN is None else torch.from_numpy(np.ascontiguousarray(CIN, np.float32))
    Ym = torch.from_numpy(np.nan_to_num(Yt, nan=0.0).astype(np.float32))
    Mk = torch.from_numpy((~np.isnan(Y)).astype(np.float32))
    kinds = [o["activation"] for o in outs]

    def loss_of(idx):
        ii = torch.from_numpy(idx)
        z = model(Vt[ii], None if Ct is None else Ct[ii])
        y, m = Ym[ii], Mk[ii]
        tot = 0.0
        for k, a in enumerate(kinds):
            if a == "sigmoid":
                l = torch.nn.functional.binary_cross_entropy_with_logits(z[:, k], y[:, k], reduction="none")
            elif a == "tanh":
                l = (torch.tanh(z[:, k]) - y[:, k]) ** 2
            else:
                l = torch.nn.functional.huber_loss(z[:, k], y[:, k], reduction="none", delta=1.0)
            tot = tot + (l * m[:, k]).sum() / m[:, k].sum().clamp(min=1.0)
        return tot

    rng = np.random.default_rng(0)
    best, best_state, bad, ep = math.inf, None, 0, 0
    for ep in range(1, tpl["epochs"] + 1):
        model.train()
        perm = tr[rng.permutation(len(tr))]
        for s in range(0, len(perm), tpl["batch"]):
            opt.zero_grad()
            loss_of(perm[s:s + tpl["batch"]]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(sum(loss_of(va[s:s + 8192]) * len(va[s:s + 8192]) for s in range(0, len(va), 8192)) / max(len(va), 1))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tpl["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, ymu, ysd, ep


def predict_z(model, V: np.ndarray, CIN, idx: np.ndarray) -> np.ndarray:
    torch = _torch()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), 4096):
            ii = idx[s:s + 4096]
            v = torch.from_numpy(np.ascontiguousarray(V[ii], np.float32))
            c = None if CIN is None else torch.from_numpy(np.ascontiguousarray(CIN[ii], np.float32))
            out.append(model(v, c).numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, 0), np.float32)


# ---------------------------------------------------------------------------------------------------------- export


class _Blob:
    def __init__(self):
        self.parts: list[np.ndarray] = []
        self.size = 0

    def put(self, a) -> int:
        a = np.asarray(a, np.float32).reshape(-1)
        off = self.size
        self.parts.append(a)
        self.size += a.size
        return off

    def bytes(self) -> bytes:
        return (np.concatenate(self.parts) if self.parts else np.zeros(0, np.float32)).astype("<f4").tobytes()


def export(name: str, tpl_name: str, tpl: dict, model, inputs: dict, outs: list[dict], vmu, vsd, fmu, fsd, ymu, ysd, enc) -> tuple[dict, bytes]:
    """The duck-sense/1 spec + bin. fmu / fsd: the emb1 features' training standardisation (folded into mlp[0])."""
    b = _Blob()
    spec: dict = {"format": C.FORMAT, "name": name}
    if tpl["conv"] is not None:
        spec["picture"] = C.PICTURE
    spec["inputs"] = {"body": list(inputs["body"]), "cam": list(inputs["cam"]), "prev": bool(inputs["prev"]), "prev_out": False}
    n_conv = 0
    if tpl["conv"] == "emb1":
        block, parts = enc.camera_block(0)
        for p in parts:
            b.put(p)  # camera_block lays its parts out from offset 0, in this order
        spec["conv"] = {"convs": block["convs"], "dense": block["dense"]}
        n_conv = block["dense"]["out"]
    elif tpl["conv"] == "train":
        convs = []
        mods = [m for m in model.conv if hasattr(m, "weight")]
        for (cin, cout, k, s, p), m in zip(tpl["convs"], mods):
            convs.append({"in": cin, "out": cout, "k": k, "stride": s, "pad": p, "weight_offset": b.put(m.weight.detach().numpy()),
                          "bias_offset": b.put(m.bias.detach().numpy()), "activation": "elu"})
        dn = model.dense
        dense = {"in": dn.in_features, "out": dn.out_features, "weight_offset": b.put(dn.weight.detach().numpy()),
                 "bias_offset": b.put(dn.bias.detach().numpy()), "activation": "elu"}
        spec["conv"] = {"convs": convs, "dense": dense}
        n_conv = dn.out_features
    spec["norm"] = {"mean_offset": b.put(vmu), "std_offset": b.put(vsd)}
    mlp = []
    L = len(model.mlp)
    for k, l in enumerate(model.mlp):
        W = l.weight.detach().numpy().astype(np.float64)
        bb = l.bias.detach().numpy().astype(np.float64)
        if k == 0 and fmu is not None:  # W_f ((f - mu) / sd) = (W_f / sd) f - (W_f / sd) mu
            Wf = W[:, :n_conv] / fsd[None, :]
            bb = bb - Wf @ fmu
            W = np.concatenate([Wf, W[:, n_conv:]], 1)
        if k == L - 1:  # linear outputs: z * sd + mu
            W = W * ysd[:, None]
            bb = bb * ysd + ymu
        mlp.append({"in": W.shape[1], "out": W.shape[0], "weight_offset": b.put(W), "bias_offset": b.put(bb),
                    "activation": tpl["act"] if k < L - 1 else "none"})
    spec["mlp"] = mlp
    spec["outputs"] = [{**o, "none": o["none"] if isinstance(o["none"], str) else float(o["none"])} for o in outs]
    return spec, b.bytes()


# ---------------------------------------------------------------------------------------------------------- parity


def parity(spec: dict, blob: bytes, d: Data, torch_out, test_mask: np.ndarray, inputs: dict, max_rows: int = 400) -> dict:
    """The written file through ref.py vs the torch forward pass (torch_out(seq rows, their pictures | None) -> outputs) on
    held-out sequences, each fed from its start (so prev matches)."""
    ref = RefSense(spec, np.frombuffer(blob, "<f4"))
    body, cam = inputs["body"], inputs["cam"]
    widths = [C.body_width(n) for n in body]
    cur = d.current(body, cam)
    sees = spec.get("conv") is not None or bool(spec.get("ops"))
    rows, worst = 0, 0.0
    for s in (s for s in d.seqs if test_mask[s[0]]):
        s = s[: max(1, max_rows - rows)]
        ref.reset()
        pics = d.pictures(s) if sees else None
        got = []
        for k, i in enumerate(s):
            c = cur[i]
            bd, at = {}, 0
            for n, w in zip(body, widths):
                bd[n] = [float(x) for x in c[at:at + w]] if n in C.BODY_GROUPS else float(c[at])
                at += w
            got.append(ref.step(None if pics is None else pics[k], bd, {n: float(c[at + q]) for q, n in enumerate(cam)}))
        worst = max(worst, float(np.abs(np.asarray(got) - torch_out(s, pics)).max()))
        rows += len(s)
        if rows >= max_rows:
            break
    return {"rows": rows, "max_abs_diff": float(f"{worst:.3g}"), "tol": PARITY_TOL, "ok": worst <= PARITY_TOL}


# ---------------------------------------------------------------------------------------------------------- the command


def data_warnings(d: Data, current=None) -> list[str]:
    """Kit 1.28.0: the collect dirs disagree on the camera world or the sim, or a dir's camera world is not the
    Challenge's current one (`current(challenge_id, picture)`, default bridge.camera_world): a sense trained there may
    read a different picture than a run now shows. Warnings only; [] when all agree (or a dir records none)."""
    if current is None:
        from ..bridge import camera_world as current
    out = []
    for field, what in (("camera_world", "camera world"), ("sim_version", "sim")):
        by_ch: dict[str, dict[str, list[str]]] = {}
        for dd in d.dirs:
            v = getattr(dd, field)
            if v is not None:
                by_ch.setdefault(dd.challenge if field == "camera_world" else "", {}).setdefault(v, []).append(dd.path.name)
        for ch, vals in by_ch.items():
            if len(vals) > 1:
                out.append(f"the data dirs{' of ' + ch if ch else ''} disagree on the {what}: "
                           + "; ".join(f"{', '.join(n)}: {v}" for v, n in vals.items()) + " (re-collect them on one)")
    for dd in d.dirs:
        if dd.camera_world is None:
            continue
        now = current(dd.challenge, dd.picture_id or C.PICTURE)
        if now is not None and now != dd.camera_world:
            out.append(f"{dd.path.name}: collected on camera world {dd.camera_world}, {dd.challenge}'s is {now} now "
                       f"(what the camera sees changed): re-collect it")
    return out


def train(data_dirs: list[str], label_src: str, label_where: str, template: str, out: str, *, hidden=None, inputs=None,
          outputs=None, split_seed: int = 0, ducks: str = "own", log=print) -> dict:
    """Train one sense and write <out>.sense.json + .sense.bin. Returns the eval dict. ducks: see run_label."""
    if template not in TEMPLATES:
        raise SystemExit(f"--template must be one of {', '.join(TEMPLATES)}")
    tpl = dict(TEMPLATES[template])
    if hidden:
        tpl["hidden"] = list(hidden)
    outp = Path(out).expanduser()
    name = outp.name.removesuffix(".sense.json")
    if not C.NAME_RE.match(name):
        raise SystemExit(f"--out name {name!r} must match {C.NAME_RE.pattern} (it is the sense's name and default `as`)")
    t0 = time.perf_counter()
    d = Data(data_dirs)
    warns = data_warnings(d)
    for w in warns:
        log(f"warning: {w}")
    ns = load_label(label_src, label_where)
    declared_out = outputs if outputs is not None else ns.get("OUTPUTS")
    inp = dict(ns.get("INPUTS") or {})
    if inputs:
        inp.update(inputs)
    body = inp.get("body") or [b for b in DEFAULT_BODY if b in d.body_names()]
    cam = inp.get("cam") if inp.get("cam") is not None else [c for c in d.cam_names() if c in C.CAM_INPUTS]
    for n in body:
        if n not in C.BODY_INPUTS:
            raise SystemExit(f"body input {n!r} is not in the sensor contract ({', '.join(C.BODY_INPUTS)})")
    for n in cam:
        if n not in C.CAM_INPUTS:
            raise SystemExit(f"cam input {n!r} is not a sense input ({', '.join(C.CAM_INPUTS)})")
    inputs = {"body": body, "cam": cam, "prev": bool(inp.get("prev", True))}
    if tpl["conv"] is not None:
        for dd in d.dirs:
            if dd.picture_id not in (None, C.PICTURE):
                raise SystemExit(f"{dd.path}: picture {dd.picture_id!r}; a picture template needs {C.PICTURE} (re-collect)")
    names, Y = run_label(ns, d, declared_out, ducks)
    outs = output_specs(names, Y, declared_out)
    t_label = time.perf_counter() - t0
    has = ~np.isnan(Y).all(1)
    trm, vam, tem = split(d, split_seed)
    tr, va, te = (np.flatnonzero(m & has) for m in (trm, vam, tem))
    if not len(tr) or not len(va) or not len(te):
        raise SystemExit(f"labelled rows: train {len(tr)}, val {len(va)}, test {len(te)}; every split needs some")
    X = d.vector(body, cam, inputs["prev"])
    vmu = X[tr].mean(0)
    vsd = X[tr].std(0)
    vmu, vsd = vmu.astype(np.float32), vsd.astype(np.float32)
    vsd = np.where(vsd >= np.float32(C.STD_MIN), vsd, np.float32(1.0))  # every std >= SENSE_STD_MIN, as float32
    V = ((X - vmu.astype(np.float64)) / vsd.astype(np.float64)).astype(np.float32)
    enc, fmu, fsd = None, None, None
    used = np.concatenate([tr, va, te])
    V_fit, CIN_fit = V, None
    if tpl["conv"] == "emb1":  # the frozen encoder's 64 numbers, once per row (numpy, cnn.ts-exact), standardised, first in x
        from .. import encoder as E
        enc = E.load("emb1")
        F = np.zeros((d.n, enc.latent_dim), np.float32)
        for s in range(0, len(used), 2048):
            ii = np.sort(used[s:s + 2048])
            F[ii] = enc.latent(d.pictures(ii))
        fmu = F[tr].astype(np.float64).mean(0)
        fsd = F[tr].astype(np.float64).std(0)
        fsd = np.where(fsd > 1e-6, fsd, 1.0)
        V_fit = np.concatenate([((F - fmu) / fsd).astype(np.float32), V], 1)
    elif tpl["conv"] == "train":
        CIN_fit = np.zeros((d.n, 4, 16, 32), np.float32)
        CIN_fit[used] = d.pictures(used)
    t_prep = time.perf_counter() - t0 - t_label
    t1 = time.perf_counter()
    model, ymu, ysd, epochs = fit(tpl, V_fit, CIN_fit, Y, outs, tr, va, log)
    t_fit = time.perf_counter() - t1
    z = predict_z(model, V_fit, CIN_fit, te)
    P_te = _outputs_from_z(z, outs, ymu, ysd)
    base = np.array([np.nanmean(Y[tr, k]) for k in range(len(outs))])
    ev = {"template": template, "metric": "held-out seeds: AUC for sigmoid outputs, MAE otherwise",
          "test": metrics(outs, Y[te], P_te, base)}
    groups = d.group_names()
    if len(d.dirs) > 1:
        ev["per_world"] = {}
        for k, g in enumerate(groups):
            m = d.dir_of[te] == k
            if m.any():
                ev["per_world"][g] = metrics(outs, Y[te][m], P_te[m], base)
    spec, blob = export(name, template, tpl, model, inputs, outs, vmu, vsd, fmu, fsd, ymu, ysd, enc)
    spec["definition"] = {"label": label_src, "collect": "; ".join(dd.collect_command() for dd in d.dirs)}
    if tpl["conv"] == "emb1":
        def torch_out(s, pics):
            f = ((enc.latent(pics).astype(np.float64) - fmu) / fsd).astype(np.float32)
            return _outputs_from_z(predict_z(model, np.concatenate([f, V[s]], 1), None, np.arange(len(s))), outs, ymu, ysd)
    elif tpl["conv"] == "train":
        def torch_out(s, pics):
            return _outputs_from_z(predict_z(model, V[s], pics, np.arange(len(s))), outs, ymu, ysd)
    else:
        def torch_out(s, pics):
            return _outputs_from_z(predict_z(model, V[s], None, np.arange(len(s))), outs, ymu, ysd)
    par = parity(spec, blob, d, torch_out, tem, inputs)
    ev["parity"] = par
    ev["rows"] = {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te)), "all": int(d.n)}
    ev["test_seeds"] = sorted({f"{groups[a]}:{b}" for a, b in zip(d.dir_of[te].tolist(), d.seed[te].tolist())})
    ev["fit_s"] = round(t_fit, 2)
    ev["prep_s"] = round(t_label + t_prep, 2)
    ev["epochs"] = int(epochs)
    ev["macs"] = C.macs(spec)
    from .. import KIT_VERSION
    ev["train"] = {"template": template, "hidden": tpl["hidden"], "act": tpl["act"], "split_seed": split_seed,
                   "ducks": ducks, "data": [dd.path.name for dd in d.dirs], "label_file": Path(label_where).name,
                   "camera_world": [dd.camera_world for dd in d.dirs], "sim": [dd.sim_version for dd in d.dirs], "kit": KIT_VERSION,
                   "outputs_declared": declared_out is not None}
    spec["eval"] = ev
    outp.parent.mkdir(parents=True, exist_ok=True)
    jp = outp.with_name(f"{name}.sense.json")
    jb = (json.dumps(spec, indent=1) + "\n").encode()
    jp.write_bytes(jb)
    jp.with_name(f"{name}.sense.bin").write_bytes(blob)
    p = C.problem(spec, len(blob)) or C.value_problem(spec, np.frombuffer(blob, "<f4"))
    if p is not None:
        raise SystemExit(f"{jp}: the written sense is invalid: {p}")
    ev["_files"] = {"json": str(jp), "id": C.sense_id(json.loads(jb), blob)}
    ev["_warnings"] = warns
    return ev


def refit(sense: str, data_dirs: list[str], out: str | None = None, log=print) -> dict:
    """Retrain a sense's template, inputs, outputs and label on new data (e.g. after a new picture)."""
    jp, _ = C.paths(sense)
    j = json.loads(jp.read_text())
    tr = (j.get("eval") or {}).get("train") or {}
    if "template" not in tr or "definition" not in j:
        raise SystemExit(f"{jp}: no eval.train / definition (not written by `duck_rl.sense train`); train it again instead")
    outputs = {o["name"]: {k: o[k] for k in ("activation", "unit", "min", "max", "none")} for o in j["outputs"]}
    inputs = {"body": j["inputs"]["body"], "cam": j["inputs"]["cam"], "prev": j["inputs"]["prev"]}
    target = out or str(jp.with_name(j["name"]))
    return train(data_dirs, j["definition"]["label"], tr.get("label_file", "<definition.label>"), tr["template"], target,
                 hidden=tr.get("hidden"), inputs=inputs, outputs=outputs, split_seed=int(tr.get("split_seed", 0)),
                 ducks=tr.get("ducks", "all"), log=log)  # written before 2026-11-16: every duck's rows
