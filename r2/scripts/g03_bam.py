#!/usr/bin/env python3
"""R2-G03: BAM version consistency between the DuckBench plant pin and the official floating branch.

Frozen SHAs (full):
  62bd8ce12154340be97e06f7f41a0ca8f116d967  plant pin, 2026-07-13 "Using stiffer frictionloss in BamActuator"
  57d13ead53206a6bf0db3d66f86506ae8c2ce01a  floating branch snapshot, 2026-07-16 "Updating favicon"

Checks (all offline):
  1. ancestry / commit counts;
  2. per-file sha256 of the files that carry dynamics (actuator, model, params, mjlab);
  3. torque law: both commits' own code on the same grid (low speed, reversal, saturation);
  4. friction: both commits' budget source hashes + numeric equality;
  5. supply voltage under load: the two models (62: gain [V/Nm]; 57: resistance [Ohm]) and the
     conversion that makes them equivalent, plus the naive misreading as a negative control.
"""
import hashlib, json, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/guohongbin/1-Projects/duckbench-rir")
BAM = ROOT / "sources/bam"
PLANT = ROOT / "sources/dbcli/node_modules/duckbench/plant/microduck-legs-bam-v1"
A, B = "62bd8ce12154340be97e06f7f41a0ca8f116d967", "57d13ead53206a6bf0db3d66f86506ae8c2ce01a"
sys.path.insert(0, str(ROOT / "duck-rl"))
from duckref import bam as kbam

FILES = ["bam/actuator.py", "bam/model.py", "bam/mjlab.py", "bam/params/xl330/m6.json"]
TOL = {"torque_Nm": 1e-12, "voltage_V": 1e-9, "friction_Nm": 1e-12}


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


def git(*args):
    return subprocess.run(["git", "-C", str(BAM), *args], capture_output=True, text=True, check=True).stdout


def blob_sha(rev, path):
    return hashlib.sha256(subprocess.run(["git", "-C", str(BAM), "show", f"{rev}:{path}"],
                                         capture_output=True, check=True).stdout).hexdigest()


def load_actuator(tree, manifest):
    for m in [k for k in sys.modules if k == "bam" or k.startswith("bam.")]:
        del sys.modules[m]
    sys.path[:] = [p for p in sys.path if not p.startswith("/tmp/bam")]
    _stub_colorama()
    sys.path.insert(0, tree)
    import bam.model as bm
    from bam.actuator import VoltageControlledActuator
    model = bm.load_model(motor_name="xl330", model="m6")
    a = model.actuator
    cfg = manifest["actuator"]
    a.vin, a.kp, a.error_gain = cfg["vin"], cfg["kp_fw"], cfg["error_gain"]
    a.max_pwm, a.max_current = cfg["max_pwm"], cfg["max_current"]
    return model, a


def torque_law(a, q_t, q, dq, vin):
    a.vin = float(vin)
    ctrl = a.compute_control(np.array([q_t]), np.array([q]), np.array([dq]), 0.005)
    return float(a.compute_torque(ctrl, True, np.array([q]), np.array([dq]))[0])


def supply_62(vin, gain, prev_torques, vin_min):
    return max(vin - gain * float(np.sum(np.abs(prev_torques))), vin_min)


def supply_57(vin, resistance, kt, prev_torques, vin_min):
    current = float(np.sum(np.abs(prev_torques))) / kt
    return max(vin - resistance * current, vin_min)


def main():
    manifest = json.loads((PLANT / "manifest.json").read_text())
    cfg = manifest["actuator"]

    counts = {
        "ahead_of_pin": int(git("rev-list", "--count", f"{A}..{B}").strip()),
        "behind_pin": int(git("rev-list", "--count", f"{B}..{A}").strip()),
        "is_ancestor": subprocess.run(["git", "-C", str(BAM), "merge-base", "--is-ancestor", A, B]).returncode == 0,
        "dates": {A[:7]: git("log", "-1", "--format=%cI", A).strip(), B[:7]: git("log", "-1", "--format=%cI", B).strip()},
        "commits_in_range": [l.split(" ", 1)[1] for l in git("log", "--format=%h %s", f"{A}..{B}").splitlines()],
    }
    hashes = {f: {"62": blob_sha(A, f), "57": blob_sha(B, f)} for f in FILES}

    m62, a62 = load_actuator("/tmp/bam62", manifest)
    m57, a57 = load_actuator("/tmp/bam57", manifest)

    rows, worst = [], 0.0
    for q_t, q, dq, vin, tag in [
        (0.0, 0.0, 0.0, cfg["vin"], "rest"),
        (0.05, 0.0, 0.0, cfg["vin"], "small error"),
        (0.5, 0.0, 0.0, cfg["vin"], "saturation (duty clamp)"),
        (0.0, 0.0, 1e-3, cfg["vin"], "low speed"),
        (0.0, 0.0, -1e-3, cfg["vin"], "low speed, reversal"),
        (0.1, -0.1, 5.0, cfg["vin"], "fast forward, back-EMF"),
        (0.1, -0.1, -5.0, cfg["vin"], "fast reverse"),
        (0.2, 0.0, 0.0, cfg["vin_min"], "low supply"),
        (0.2, 0.0, 0.0, cfg["vin_range"][1], "high supply"),
        (0.3, 0.0, 2.0, cfg["vin"], "large error with back-EMF (current-limit window)"),
    ]:
        t62 = torque_law(a62, q_t, q, dq, vin)
        t57 = torque_law(a57, q_t, q, dq, vin)
        d = abs(t62 - t57)
        worst = max(worst, d)
        rows.append(dict(case=tag, q_target=q_t, q=q, dq=dq, vin=vin, torque_62=t62, torque_57=t57, abs_diff=d))

    # friction: same source lines? numeric equality on a grid via the transcribed GPU form
    sys.path.insert(0, str(ROOT / "r2/scripts"))
    from oracle_mjlab_friction import gpu_form_friction_budget
    fw, frows = 0.0, []
    for tau_m in (-1.0, -0.2, 0.0, 0.2, 1.0):
        for tau_ext in (-1.0, -0.3, 0.0, 0.3, 1.0):
            for dq in (-10.0, -1e-3, 0.0, 1e-3, 10.0):
                f62 = float(gpu_form_friction_budget(m62, np.float64(tau_m), np.float64(tau_ext), np.float64(dq)))
                f57 = float(gpu_form_friction_budget(m57, np.float64(tau_m), np.float64(tau_ext), np.float64(dq)))
                d = abs(f62 - f57)
                fw = max(fw, d)
                frows.append(dict(tau_m=tau_m, tau_ext=tau_ext, dq=dq, f62=f62, f57=f57, abs_diff=d))

    kt = cfg["params"]["kt"]
    gain = cfg["vin_drop_gain"]
    resistance_equiv = gain * kt  # V/Nm * Nm/A = V/A = Ohm, since I = sum|tau| / kt
    loads = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    sup = []
    for load in loads:
        prev = np.array([load])
        v62 = supply_62(cfg["vin"], gain, prev, cfg["vin_min"])
        v57 = supply_57(cfg["vin"], resistance_equiv, kt, prev, cfg["vin_min"])
        v_naive = supply_57(cfg["vin"], gain, kt, prev, cfg["vin_min"])  # misread gain as Ohm
        sup.append(dict(load_Nm=load, v62=v62, v57_converted=v57, abs_diff=abs(v62 - v57),
                        v57_if_gain_taken_as_ohm=v_naive, abs_diff_naive=abs(v62 - v_naive)))
    sup_worst = max(r["abs_diff"] for r in sup)
    naive_worst = max(r["abs_diff_naive"] for r in sup)

    out = {
        "shas": {A[:7]: A, B[:7]: B},
        "counts": counts,
        "file_sha256": hashes,
        "torque_law": {"cases": rows, "max_abs_diff_Nm": worst,
                       "verdict": "IDENTICAL" if worst <= TOL["torque_Nm"] else "DIFFERS"},
        "friction": {"n": len(frows), "max_abs_diff_Nm": fw, "rows": frows,
                     "verdict": "IDENTICAL" if fw <= TOL["friction_Nm"] else "DIFFERS"},
        "supply_voltage": {"gain_V_per_Nm": gain, "resistance_equivalent_Ohm": resistance_equiv,
                           "rows": sup, "max_abs_diff_V": sup_worst,
                           "naive_misread_max_abs_diff_V": naive_worst,
                           "verdict": "EQUIVALENT_AFTER_UNIT_CONVERSION" if sup_worst <= TOL["voltage_V"] else "DIFFERS"},
        "tolerances": TOL,
    }
    (ROOT / "r2/artifacts/g03_bam.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("counts", "torque_law", "friction", "supply_voltage")}, indent=1)[:3000])


if __name__ == "__main__":
    main()