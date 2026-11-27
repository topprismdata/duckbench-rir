#!/usr/bin/env python3
"""R2-G02: head-command EMA — semantics, timing, and a paired with/without-EMA run.

The DuckBench plant smooths the 4 head command slots ([51:55]) with an EMA while the
official training/Arena/robot links feed the command directly. This script:

  1. freezes the four slots' meaning from the plant manifest's policy command names and
     GUIDE.md's raw 13-slot order;
  2. measures the EMA step response against the analytic y_n = target*(1-(1-k)^n);
  3. runs a paired closed-loop comparison (same policy weights, same start, same command,
     same seed): A = stock EMA, B = EMA bypassed by setting the module constant to 1.0 in
     an isolated copy of the module (the frozen plant files are not edited).

Policy: submissions/e1_s2 (the best E1 run), chosen by a rule frozen before the runs:
highest E1 best-half. It is not selected on the EMA outcome.
"""
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/guohongbin/1-Projects/duckbench-rir")
PLANT = ROOT / "sources/dbcli/node_modules/duckbench/plant/microduck-legs-bam-v1"
POLICY = ROOT / "duck-rl/submissions/e1_s2"
sys.path.insert(0, str(ROOT / "duck-rl"))

import duckref.plant as plant_mod
from duckref.bam import DuckRefBam

HEAD_STEP = np.array([0.0, 0.72, 0.0, 0.0])  # head_pitch 0.72 rad: the leaderboard policy's value
TICKS = 400                                  # 8 s at 50 Hz


def load_mlp(dirpath):
    meta = json.loads((dirpath / "policy.mlp.json").read_text())
    blob = np.fromfile(dirpath / "policy.mlp.bin", dtype="<f4")
    nrm = meta["norm"]
    sub = blob[nrm["sub_offset"]:nrm["sub_offset"] + nrm["len"]]
    div = blob[nrm["div_offset"]:nrm["div_offset"] + nrm["len"]]
    layers = []
    for L in meta["layers"]:
        w = blob[L["weight_offset"]:L["weight_offset"] + L["in"] * L["out"]].reshape(L["out"], L["in"])
        b = blob[L["bias_offset"]:L["bias_offset"] + L["out"]]
        layers.append((w.copy(), b.copy(), L["activation"], L.get("elu_alpha", 1.0)))
    assert meta["input_dim"] == 61 and meta["output_dim"] == 14

    def run(obs):
        x = (np.asarray(obs, dtype=np.float32).reshape(-1) - sub) / div
        for i, (w, b, act, alpha) in enumerate(layers):
            x = w @ x + b
            if act == "elu":
                x = np.where(x > 0, x, alpha * (np.expm1(x))).astype(np.float32)
            elif act == "tanh":
                x = np.tanh(x)
            elif act == "relu":
                x = np.maximum(x, 0)
        return x
    return run, meta


class Runner:
    """Closed-loop roll-out of a duck-mlp policy on the plant v1 reference."""

    def __init__(self, policy_fn):
        self.ref = DuckRefBam(bundle_dir=str(PLANT))
        self.ref.reset()
        self.policy = policy_fn

    def run(self, ticks, head_target):
        rows = []
        for k in range(ticks):
            d = self.ref.data
            self.ref.data.qpos  # noqa: B018 (kept explicit: state is read by build_obs)
            mj_forward = __import__("mujoco").mj_forward
            mj_forward(self.ref.model, d)
            obs, parts = self.ref.build_obs("alpha_walking", (0.19, 0.0, 0.0), head_target, None)
            action = self.policy(obs).astype(np.float32)
            self.ref.bam.latch(self.ref.default_pos + action.astype(np.float64))
            for _ in range(self.ref.decimation):
                self.ref.bam.before_substep(d)
                __import__("mujoco").mj_step(self.ref.model, d)
            self.ref.prev_action = action
            self.ref.lag_joint_vel = parts["vel_now"]
            rows.append(dict(k=k, head=obs[51:55].copy(), action=action.copy(),
                             x=float(d.qpos[0]), y=float(d.qpos[1]), z=float(d.qpos[2]),
                             tilt=self.ref.tilt_deg() if hasattr(self.ref, "tilt_deg") else None))
        return rows


def summarize(rows):
    x = np.array([r["x"] for r in rows])
    return dict(final_x=float(x[-1]), max_x=float(x.max()),
                finish_tick=int(np.argmax(x >= 4.5)) if (x >= 4.5).any() else None,
                n=len(rows))


def main():
    names = plant_mod.DuckRef(PLANT).manifest["policies"]["alpha_walking"]["command_names"]
    head_slots = "neck_pitch head_pitch head_yaw head_roll".split()
    assert names == ["twist", "head_pose", "body_pose"]
    k_ema = plant_mod.HEAD_EMA

    # 1. analytic step-response check on the observation builder alone
    ref = DuckRefBam(bundle_dir=str(PLANT))
    ref.reset()
    measured, analytic = [], []
    z = 0.0
    for n in range(1, 11):
        obs, _ = ref.build_obs("alpha_walking", (0, 0, 0), HEAD_STEP, None)
        z = z + k_ema * (HEAD_STEP[1] - z)
        measured.append(float(obs[51 + 1]))
        analytic.append(float(z))
    step_err = float(np.max(np.abs(np.array(measured) - np.array(analytic))))

    # 2. paired closed-loop runs (primary head step + a sensitivity ladder incl. 0)
    policy, meta = load_mlp(POLICY)
    ladder = {}
    for hp in (0.0, 0.2, 0.6, 0.72):
        step = np.array([0.0, hp, 0.0, 0.0])
        ra = Runner(policy).run(TICKS, step)
        saved = plant_mod.HEAD_EMA
        plant_mod.HEAD_EMA = 1.0  # isolated override: y <- target (no smoothing)
        try:
            rb = Runner(policy).run(TICKS, step)
        finally:
            plant_mod.HEAD_EMA = saved
        sa, sb = summarize(ra), summarize(rb)
        ladder[hp] = {
            "finish_tick_A_ema": sa["finish_tick"], "finish_tick_B_no_ema": sb["finish_tick"],
            "final_x_A": sa["final_x"], "final_x_B": sb["final_x"],
            "obs_head_max_abs_diff_rad": float(np.max(np.abs(np.array([r["head"] for r in ra]) - np.array([r["head"] for r in rb])))),
            "action_max_abs_diff": float(np.max(np.abs(np.array([r["action"] for r in ra]) - np.array([r["action"] for r in rb])))),
            "torso_max_abs_diff_m": float(np.max(np.abs(np.array([[r["x"], r["y"], r["z"]] for r in ra]) - np.array([[r["x"], r["y"], r["z"]] for r in rb])))),
        }
    a, b = None, None
    a = Runner(policy).run(TICKS, HEAD_STEP)
    saved = plant_mod.HEAD_EMA
    plant_mod.HEAD_EMA = 1.0
    try:
        b = Runner(policy).run(TICKS, HEAD_STEP)
    finally:
        plant_mod.HEAD_EMA = saved

    ha = np.array([r["head"] for r in a]); hb = np.array([r["head"] for r in b])
    aa = np.array([r["action"] for r in a]); ab = np.array([r["action"] for r in b])
    pa = np.array([[r["x"], r["y"], r["z"]] for r in a]); pb = np.array([[r["x"], r["y"], r["z"]] for r in b])

    out = {
        "frozen_before_run": {
            "head_slots": dict(zip([48, 49, 50, 51], ["vx", "vy", "wz", "neck_pitch"])),
            "head_pose_slots_[51:55]": head_slots,
            "source": "GUIDE.md raw 13-slot order (vx vy wz neck_pitch head_pitch head_yaw head_roll body_0..5)",
            "ema_constant": k_ema, "update": "per policy tick (50 Hz)", "init": "zeros on reset",
            "policy_rule": "e1_s2 = highest E1 best-half (3.7663 s), frozen before the EMA runs",
            "head_step": HEAD_STEP.tolist(), "ticks": TICKS,
        },
        "step_response": {"measured": measured, "analytic": analytic, "max_abs_err_rad": step_err,
                          "tau_ticks_analytic": -1.0 / np.log(1 - k_ema)},
        "paired": {
            "obs_head_max_abs_diff_rad": float(np.max(np.abs(ha - hb))),
            "obs_head_mean_abs_diff_rad": float(np.mean(np.abs(ha - hb))),
            "action_max_abs_diff": float(np.max(np.abs(aa - ab))),
            "action_mean_abs_diff": float(np.mean(np.abs(aa - ab))),
            "torso_max_abs_diff_m": float(np.max(np.abs(pa - pb))),
            "torso_final_diff_m": float(np.linalg.norm(pa[-1] - pb[-1])),
            "A_ema": summarize(a), "B_no_ema": summarize(b),
            "A_ema_effect_size": None,
        },
        "head_pitch_ladder": ladder,
        "policy": {"path": str(POLICY), "onnx_sha256": meta["source_onnx_sha256"]},
    }
    pa_fin, pb_fin = summarize(a)["final_x"], summarize(b)["final_x"]
    out["paired"]["A_ema_effect_size"] = {
        "delta_final_x_m": pa_fin - pb_fin,
        "note": "single deterministic pair; a cross-unit normalization (metres / E1 second-SD) was removed as invalid "
                "(GPT R2 review P0)",
    }
    (ROOT / "r2/artifacts/g02_ema.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("step_response",)}, indent=1))
    print(json.dumps(out["paired"], indent=1))


if __name__ == "__main__":
    main()