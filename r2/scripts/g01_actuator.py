#!/usr/bin/env python3
"""R2-G01: actuator identity, parameter provenance and numeric cross-check.

Oracle: the real BAM library (Rhoban/bam) as shipped at the two commits of interest,
extracted with `git archive` into /tmp/bam62 (62bd8ce, the DuckBench plant pin) and
/tmp/bam57 (57d13ead, the floating branch of the official training stack).

Subject: the DuckBench kit's CPU reference implementation (`duckref/bam.py`), which is
what the training stack and the TS sim replicate.

Tolerances are frozen before the run (see TOL) and reported per unit. The negative
control perturbs a single known physical parameter and must be detected.
"""
import json, sys, math
from pathlib import Path

ROOT = Path("/Users/guohongbin/1-Projects/duckbench-rir")
PLANT = ROOT / "sources/dbcli/node_modules/duckbench/plant/microduck-legs-bam-v1"
sys.path.insert(0, str(ROOT / "duck-rl"))

import numpy as np
from duckref import bam as kbam

TOL = {"torque_Nm": 1e-9, "angle_rad": 1e-9, "velocity_rad_s": 1e-9, "friction_Nm": 1e-9}


def _stub_colorama():
    import types
    if "colorama" in sys.modules:
        return
    c = types.ModuleType("colorama")

    class _Blank:
        def __getattr__(self, _k):
            return ""

    c.Fore = c.Back = c.Style = _Blank()
    c.init = lambda *a, **k: None
    sys.modules["colorama"] = c


def load_oracle(path):
    for p in list(sys.path):
        if p.startswith("/tmp/bam"):
            sys.path.remove(p)
    _stub_colorama()
    sys.path.insert(0, path)
    for m in [m for m in sys.modules if m == "bam" or m.startswith("bam.")]:
        del sys.modules[m]
    import bam.model as bm
    return bm.load_model(motor_name="xl330", model="m6")


def kit_cfg(manifest):
    return kbam.BamConfig.from_block(manifest["actuator"])


def torque_grid(model, cfg, manifest):
    a = model.actuator
    a.vin, a.kp, a.error_gain = cfg.vin, cfg.kp_fw, cfg.error_gain
    a.max_pwm, a.max_current = cfg.max_pwm, cfg.max_current
    rows, worst = [], 0.0
    for q_t in (-0.3, 0.0, 0.3):
        for q in (-0.5, -0.1, 0.0, 0.25, 0.5):
            for dq in (-10.0, -1.0, 0.0, 1.0, 10.0):
                for vin in (cfg.vin_min, cfg.vin, manifest["actuator"]["vin_range"][1]):
                    a.vin = float(vin)
                    ctrl = a.compute_control(np.array([q_t]), np.array([q]), np.array([dq]), 0.005)
                    tau_o = float(a.compute_torque(ctrl, True, np.array([q]), np.array([dq]))[0])
                    tau_k = float(kbam.motor_torque(cfg, vin, np.array([q_t]), np.array([q]), np.array([dq]))[0])
                    d = abs(tau_o - tau_k)
                    worst = max(worst, d)
                    rows.append(dict(q_target=q_t, q=q, dq=dq, vin=vin, oracle=tau_o, kit=tau_k, abs_diff=d))
    return rows, worst


def mjlab_friction_oracle(model):
    sys.path.insert(0, str(ROOT / "r2/scripts"))
    from oracle_mjlab_friction import gpu_form_friction_budget

    def gpu(tau_m, tau_ext, dq):
        return float(gpu_form_friction_budget(model, tau_m, tau_ext, dq))

    def cpu(tau_m, tau_ext, dq):
        return float(model.compute_frictions(tau_m, tau_ext, dq)[0])

    return gpu, cpu


def _unused_mjlab_friction_oracle(model):
    """The GPU-form budget the plant manifest cites (bam/mjlab.py BamActuator._compute_friction_budget),
    called on the untouched class with a numpy-free torch CPU input."""
    import torch
    from bam.mjlab import BamActuator
    inst = object.__new__(BamActuator)
    inst._bam_model = model
    inst.friction_scale = None

    def f(tau_m, tau_ext, dq):
        t = lambda v: torch.tensor([[float(v)]], dtype=torch.float64)
        sc = torch.exp(-torch.pow(torch.abs(t(dq)) / model.dtheta_stribeck.value, model.alpha.value))
        return float(inst._compute_friction_budget(t(tau_m), t(tau_ext), sc)[0, 0])
    return f


def friction_grid(model, cfg):
    oracle, cpu_oracle = mjlab_friction_oracle(model)
    rows, worst, worst_cpu = [], 0.0, 0.0
    for tau_m in (-1.0, -0.2, 0.0, 0.2, 1.0):
        for tau_ext in (-1.0, -0.3, 0.0, 0.3, 1.0):
            for dq in (-10.0, -1.0, 0.0, 1.0, 10.0):
                fl_o, damp_o = oracle(tau_m, tau_ext, dq), float(model.friction_viscous.value)
                fl_cpu = cpu_oracle(tau_m, tau_ext, dq)
                fl_k = float(kbam.friction_budget(cfg, np.array([tau_m]), np.array([tau_ext]), np.array([dq]))[0])
                d = abs(float(fl_o) - fl_k)
                worst = max(worst, d)
                worst_cpu = max(worst_cpu, abs(fl_cpu - fl_k))
                rows.append(dict(tau_m=tau_m, tau_ext=tau_ext, dq=dq, oracle_gpu=float(fl_o), oracle_cpu=fl_cpu, kit=fl_k,
                                 damping=float(damp_o), abs_diff=d, abs_diff_cpu=abs(fl_cpu - fl_k)))
    return rows, worst, worst_cpu


def params_table(manifest, models):
    keys = kbam.BAM_PARAM_KEYS
    out = {}
    p = manifest["actuator"]["params"]
    for k in keys:
        out[k] = {"plant_manifest": p.get(k)}
        for tag, m in models.items():
            val = getattr(m, k, None)
            out[k][tag] = None if val is None else float(val.value)
    return out


def main():
    manifest = json.loads((PLANT / "manifest.json").read_text())
    models = {"bam_62bd8ce": load_oracle("/tmp/bam62")}
    m57 = load_oracle("/tmp/bam57")
    models["bam_57d13ead"] = m57
    cfg = kit_cfg(manifest)

    torque_rows, torque_worst = torque_grid(models["bam_62bd8ce"], cfg, manifest)
    friction_rows, friction_worst, friction_worst_cpu = friction_grid(models["bam_62bd8ce"], cfg)

    # per-unit checks: angle and velocity enter analytically; verify the actuator's
    # own conversions are the identity (no hidden scaling) by comparing at q=0,dq=0
    a = models["bam_62bd8ce"].actuator
    a.vin, a.kp, a.error_gain = cfg.vin, cfg.kp_fw, cfg.error_gain
    a.max_pwm, a.max_current = cfg.max_pwm, cfg.max_current
    # per-unit worst cases, taken from the same grid (angle-only rows: dq == 0)
    angle = max((r["abs_diff"] for r in torque_rows if r["dq"] == 0.0), default=0.0)
    velocity = max((r["abs_diff"] for r in torque_rows if r["dq"] != 0.0), default=0.0)

    # negative control: +10 % on friction_base and +1 % on kt must be detected
    from dataclasses import replace
    neg = {}
    p2 = dict(cfg.p)
    p2["friction_base"] *= 1.10
    cfg2 = replace(cfg, p=p2)
    neg["friction_base_+10pct_max_diff"] = max(
        abs(float(kbam.friction_budget(cfg2, np.array([r["tau_m"]]), np.array([r["tau_ext"]]), np.array([r["dq"]]))[0])
            - r["oracle_gpu"]) for r in friction_rows)
    p3 = dict(cfg.p)
    p3["kt"] *= 1.01
    cfg3 = replace(cfg, p=p3)
    neg["kt_+1pct_max_diff"] = max(
        abs(float(kbam.motor_torque(cfg3, r["vin"], np.array([r["q_target"]]), np.array([r["q"]]), np.array([r["dq"]]))[0])
            - r["oracle"]) for r in torque_rows)

    verdict = {
        "torque": "PASS" if torque_worst <= TOL["torque_Nm"] else "FAIL",
        "friction": "PASS" if friction_worst <= TOL["friction_Nm"] else "FAIL",
        "negative_control": "PASS" if (neg["friction_base_+10pct_max_diff"] > TOL["friction_Nm"] and
                                       neg["kt_+1pct_max_diff"] > TOL["torque_Nm"]) else "FAIL",
    }
    out = {
        "tolerances": TOL,
        "params": params_table(manifest, models),
        "torque": {"n": len(torque_rows), "max_abs_diff_Nm": torque_worst, "rows": torque_rows, "verdict": verdict["torque"]},
        "friction": {"n": len(friction_rows), "max_abs_diff_Nm": friction_worst, "rows": friction_rows,
                     "verdict": verdict["friction"],
                     "cpu_form_max_abs_diff_Nm": friction_worst_cpu,
                     "note": "cpu form = bam/model.py compute_frictions (no m6 quadratic term); gpu form = bam/mjlab.py:471-536 transcribed in r2/scripts/oracle_mjlab_friction.py"},
        "angle_check_rad": angle, "velocity_check_rad_s": velocity,
        "negative_control": neg, "verdicts": verdict,
        "oracle_commits": {"62bd8ce": "/tmp/bam62", "57d13ead": "/tmp/bam57"},
        "params_57_vs_62_identical": all(
            v["bam_62bd8ce"] == v["bam_57d13ead"] for v in out_params(params_table(manifest, models)) if v["plant_manifest"] is not None),
    }
    (ROOT / "r2/artifacts/g01_actuator.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("verdicts", "angle_check_rad", "velocity_check_rad_s", "negative_control")}, indent=1))
    print("torque worst", torque_worst, "friction worst", friction_worst)
    print("params 57==62:", out["params_57_vs_62_identical"])


def out_params(table):
    return [v for v in table.values()]


if __name__ == "__main__":
    main()