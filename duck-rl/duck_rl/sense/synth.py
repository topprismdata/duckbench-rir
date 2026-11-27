"""A small synthetic `duck-sense-collect/1` directory for tests and smoke runs (no sim): two ducks wander a 1 m disc,
each sees the other as a duck-coloured block in a palette picture (sky, shaded ground, the own feet), with cam_* and
body columns in the collect format and a world dict keyed as sim sensors.ts collects it (signals such as rival_dist,
Infinity as null; true_x / true_y / true_z / true_heading / true_vx / true_vy; other0_x / other0_y / other0_heading).
The numbers are made up; only the shapes, names and the picture <-> world link are real.

  python -m duck_rl.sense synth --out DIR [--seeds 0-7] [--ticks 150] [--challenge synth]
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .data import FORMAT, PICTURE_SHAPE

PAL = json.loads((Path(__file__).resolve().parents[1] / "palette.json").read_text())
FOV = 0.8  # half the horizontal field of view, rad
BODY_FIELDS = ([f"gyro_{i}" for i in range(3)] + [f"gravity_{i}" for i in range(3)] + [f"joint_pos_{i}" for i in range(14)]
               + [f"joint_vel_{i}" for i in range(14)]
               + ["t", "tilt", "upright", "fallen", "yaw_rate", "heading", "forward_speed", "side_speed", "speed"])
CAM_FIELDS = ["cam_free_left", "cam_free_ahead", "cam_free_right", "cam_free_known", "cam_duck_seen", "cam_duck_bearing",
              "cam_duck_dist", "cam_drop_left", "cam_drop_ahead", "cam_drop_right", "cam_drop_known", "cam_obstacle_ahead"]
R = 1.5  # arena radius


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _free(x: float, y: float, a: float) -> float:
    """Distance from (x, y) along heading a to the disc's edge (inf past 3 m)."""
    c, s = math.cos(a), math.sin(a)
    b = x * c + y * s
    d = -b + math.sqrt(max(b * b - (x * x + y * y - R * R), 0.0))
    return d if d < 3 else math.inf


def _picture(rng: np.random.Generator, bearing: float, dist: float) -> np.ndarray:
    _, H, W = PICTURE_SHAPE
    pic = np.zeros(PICTURE_SHAPE, np.float32)
    sky = np.array(PAL["sky"], np.float32)
    ground = np.array(PAL["colours"]["ground"], np.float32)
    duck = np.array(PAL["colours"]["duck"], np.float32)
    for r in range(H):
        if r < 5:
            pic[0, r], pic[1:, r] = 1.0, sky[:, None]
        else:
            pic[0, r] = min(1.0, 1.2 / (r - 3.5))  # lower rows see nearer ground
            pic[1:, r] = (ground * rng.uniform(0.55, 0.95))[:, None]
    if abs(bearing) < FOV and dist < 3:
        c = (0.5 - bearing / (2 * FOV)) * W  # bearing > 0 is to the left
        half = max(0.6, 2.4 / max(dist, 0.1))
        c0, c1 = max(0, int(c - half)), min(W - 1, int(c + half))
        r0, r1 = 4, min(H - 1, 6 + int(4 / max(dist, 0.25)))
        if c0 <= c1:
            pic[0, r0:r1 + 1, c0:c1 + 1] = dist / 3
            pic[1:, r0:r1 + 1, c0:c1 + 1] = (duck * rng.uniform(0.6, 0.95))[:, None, None]
    pic[1:, 14:, 13:19] = 0.04  # the own feet (self)
    pic[0, 14:, 13:19] = 0.03
    return pic


def make(out: str | Path, seeds: list[int], ticks: int = 150, challenge: str = "synth") -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pics, bodies, cams, rows, worlds = [], [], [], [], []
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        n = 2
        p = rng.uniform(-0.9, 0.9, (n, 2))
        yaw = rng.uniform(-math.pi, math.pi, n)
        v = rng.uniform(0.05, 0.25, n)
        w = np.zeros(n)
        hist = []
        for k in range(ticks):
            for i in range(n):
                to_c = math.atan2(-p[i, 1], -p[i, 0])
                pull = 2.0 * _wrap(to_c - yaw[i]) if np.hypot(*p[i]) > 1.1 else 0.0
                j = 1 - i
                to_r = math.atan2(p[j, 1] - p[i, 1], p[j, 0] - p[i, 0])
                near = np.hypot(*(p[j] - p[i])) < 0.25
                chase = (-1.5 if near else 0.7) * _wrap(to_r - yaw[i])
                w[i] = 0.7 * w[i] + 0.3 * (pull + chase + rng.normal(0, 1.0))
                v[i] = float(np.clip(v[i] + rng.normal(0, 0.03), 0.02, 0.3))
            prev = p.copy()
            yaw = np.array([_wrap(a + 0.1 * b) for a, b in zip(yaw, w)])
            p = p + 0.1 * v[:, None] * np.stack([np.cos(yaw), np.sin(yaw)], 1)
            hist.append((p.copy(), yaw.copy(), (p - prev) / 0.1, w.copy(), v.copy()))
        for i in range(n):
            j = 1 - i
            for k, (pp, yy, vel, ww, vv) in enumerate(hist):
                dx, dy = pp[j] - pp[i]
                d = float(math.hypot(dx, dy))
                b = _wrap(math.atan2(dy, dx) - yy[i])
                seen = abs(b) < FOV and d < 3
                c, s = math.cos(yy[i]), math.sin(yy[i])
                fwd, side = vel[i, 0] * c + vel[i, 1] * s, -vel[i, 0] * s + vel[i, 1] * c
                tilt = abs(rng.normal(0, 0.05))
                gyro = rng.normal(0, 0.2, 3)
                gyro[2] = ww[i]
                grav = np.array([math.sin(tilt) * 0.3, math.sin(tilt) * 0.2, -math.cos(tilt)])
                body = [*gyro, *grav, *rng.normal(0, 0.3, 14), *rng.normal(0, 1.0, 14),
                        0.1 * k, tilt, 1.0, 0.0, ww[i], _wrap(yy[i]), fwd, side, math.hypot(fwd, side)]
                cam = [_free(*pp[i], yy[i] + 0.5), _free(*pp[i], yy[i]), _free(*pp[i], yy[i] - 0.5), 1.0, float(seen),
                       b if seen else 0.0, d if seen else math.inf, math.inf, math.inf, math.inf, 0.0, math.inf]
                pics.append(_picture(rng, b, d))
                bodies.append(body)
                cams.append(cam)
                rows.append({"seed": seed, "duck": i, "tick": k, "t": round(0.1 * k, 6)})
                # as sim sensors.ts collects it: the signals (Infinity -> null), true_*, other<k>_*
                wd = {"t": round(0.1 * k, 6), "speed": float(vv[i]), "rival_dist": d, "rival_bearing": b,
                      "rival_seen": bool(abs(b) < math.pi / 2), "cam_duck_dist": d if seen else None,
                      "true_x": float(pp[i, 0]), "true_y": float(pp[i, 1]), "true_z": 0.2, "true_heading": float(yy[i]),
                      "true_vx": float(vel[i, 0]), "true_vy": float(vel[i, 1]),
                      "other0_x": float(pp[j, 0]), "other0_y": float(pp[j, 1]), "other0_heading": float(yy[j])}
                if k == ticks - 1:
                    wd["result"] = {"points": float(rng.integers(0, 3))}
                worlds.append(wd)
    np.stack(pics).astype("<f4").tofile(out / "picture.f32")
    np.asarray(bodies, "<f4").tofile(out / "body.f32")
    np.asarray(cams, "<f4").tofile(out / "cam.f32")
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "world.jsonl").write_text("".join(json.dumps(w) + "\n" for w in worlds))
    meta = {"format": FORMAT, "challenge": challenge, "strategy_id": "synthetic", "picture": "ray-rgbd-palette-4x16x32/2",
            "seeds": f"{min(seeds)}-{max(seeds)}", "n_rows": len(rows), "body_fields": BODY_FIELDS, "cam_fields": CAM_FIELDS,
            "world_fields": sorted({k for w in worlds for k in w})}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return out
