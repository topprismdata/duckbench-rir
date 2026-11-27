"""The duck's head camera in the kit, the same as the TS sim's (packages/sim/src/runtime/rayeye.ts, camera.ts, vision.ts):
the camera move's picture (RaySight: RAY_COLS x RAY_ROWS of the head camera's rays, the palette look,
"ray-rgbd-palette-4x16x32/2", docs/vision/PALETTE.md) that a frozen camera encoder reads (encoder.py, train.py ENCODER),
and HeadCamera's 64 x 64 render (the physics colours; parity tools only). The camera: the arena's head camera (the plant's
head_camera at its body offset turned by CAM_QUAT, 80 deg vertical field of view), one mj_multiRay per frame from the
camera's origin leaving out its own body (the head). The ray eye (its 12 numbers, VISION = True, the 73-input vision
moves) and the CNN eye before it were removed by LB-330 (2026-11-14).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import mujoco
import numpy as np


def njit(*jit_args, **jit_kw):
    """numba's njit, imported at the first call: a kit without numba still imports this module (only the palette
    picture's kernel needs numba)."""
    def deco(f):
        compiled = None

        def call(*args):
            nonlocal compiled
            if compiled is None:
                from numba import njit as _njit
                compiled = _njit(*jit_args, **jit_kw)(f)
            return compiled(*args)
        call.__doc__, call.__name__ = f.__doc__, f.__name__
        return call
    return deco


# the camera's square picture the rays sample (rayeye.ts RAY_RES) and the depth cap (RAY_DMAX: depth = min(d, 3) / 3)
RAY_RES, RAY_DMAX = 64, 3.0
CAM_QUAT = np.array([0.707107, 0.0, 0.0, -0.707107])
FOVY = 80.0
SKY = np.array([0.55, 0.7, 0.9])
GROUPS = np.array([1, 1, 0, 1, 0, 0], np.uint8)
# the palette picture's rays, one cast (rayeye.ts PICTURE_GROUPS, palette.json camera_groups): also the ducks' camera
# proxies (4) and the Challenges' visual-only parts (5), which a palette camera move's world keeps (duckref.plant camera_geoms)
PICTURE_GROUPS = np.array([1, 1, 0, 1, 1, 1], np.uint8)
MAT_SPECULAR, MAT_SHININESS, NO_LIGHT_AMBIENT, CUTOFF = 0.5, 64.0, 0.3, 1e6
# the camera's rays (packages/sim/src/runtime/rayeye.ts): every second column and every fourth row of the 64 x 64 pixel centres
RAY_COL_STRIDE, RAY_ROW_STRIDE = 2, 4
RAY_COLS, RAY_ROWS = RAY_RES // RAY_COL_STRIDE, RAY_RES // RAY_ROW_STRIDE
# the match world's duck name prefixes (packages/sim/src/plant.ts MATCH_DUCK_PREFIXES)
MATCH_DUCK_PREFIXES = ("", "b_", "c_", "d_", "e_", "f_", "g_", "h_")
# a camera move's picture (schema move-camera.ts, rayeye.ts refresh): depth, r, g, b per ray, [channel][row][col]
PICTURE_SHAPE = (4, RAY_ROWS, RAY_COLS)
# the picture in the shared palette (schema move-camera.ts CAMERA_PICTURE_PALETTE, palette v2); it replaced the retired
# cheap look "ray-rgbd-cheap-4x16x32/1" (colours from the physics model) on 2026-11-13
PICTURE_PALETTE = "ray-rgbd-palette-4x16x32/2"
# the cheap look's palette (docs/vision/PALETTE.md; packages/schema/src/palette.ts, generated as palette.json next to this
# file): each geom the camera sees takes its class's colour, by the same rules as packages/sim/src/runtime/geom-class.ts
PALETTE = json.loads((Path(__file__).resolve().parent / "palette.json").read_text())
PALETTE_FORMAT, PALETTE_VERSION = "duck-palette/1", 2
if PALETTE["format"] != PALETTE_FORMAT or PALETTE["version"] != PALETTE_VERSION:
    raise SystemExit(f"duck_rl/palette.json is {PALETTE['format']} v{PALETTE['version']}, this kit reads {PALETTE_FORMAT} "
                     f"v{PALETTE_VERSION}: regenerate it (node packages/schema/scripts/palette-json.ts) or update vision.py")
_RULES = [(r, re.compile(r["name"]) if "name" in r else None, re.compile(r["body"]) if "body" in r else None)
          for r in PALETTE["rules"]]


def _class_colour(c: str) -> list[float]:
    return PALETTE["colours"][c]


def _names(m: mujoco.MjModel, obj, n: int) -> list[str]:
    return [mujoco.mj_id2name(m, obj, i) or "" for i in range(n)]


def geom_classes(m: mujoco.MjModel, own_root: int = -1) -> list[str | None]:
    """Each geom's class (geom-class.ts classOfGeoms): None for a geom the camera never sees (not in camera_groups); a
    duck's geoms by duck_parts (the camera's own duck, root body own_root: self / self_accent); every other geom the first
    matching rule (on its name, body name, type, moving, and its bounding box at the rest pose), else the fallback."""
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    gnames = _names(m, mujoco.mjtObj.mjOBJ_GEOM, m.ngeom)
    bnames = _names(m, mujoco.mjtObj.mjOBJ_BODY, m.nbody)
    mnames = _names(m, mujoco.mjtObj.mjOBJ_MESH, m.nmesh)
    duck_roots = {p + "trunk_base" for p in MATCH_DUCK_PREFIXES}
    prefixes = sorted((p for p in MATCH_DUCK_PREFIXES if p), key=len, reverse=True)

    def unprefixed(nm: str) -> str:
        for p in prefixes:
            if nm.startswith(p):
                return nm[len(p):]
        return nm

    acc = PALETTE["duck_parts"]["accent"]
    out: list[str | None] = []
    for g in range(m.ngeom):
        if int(m.geom_group[g]) not in PALETTE["camera_groups"]:
            out.append(None)
            continue
        b = int(m.geom_bodyid[g])
        r = int(m.body_rootid[b])
        gtype = int(m.geom_type[g])
        mesh = mnames[int(m.geom_dataid[g])] if gtype == 7 and int(m.geom_dataid[g]) >= 0 else ""
        if r != 0 and bnames[r] in duck_roots:
            accent = unprefixed(mesh) in acc["meshes"] or unprefixed(bnames[b]) in acc["bodies"]
            out.append(("self" if r == own_root else "duck") + ("_accent" if accent else ""))
            continue
        R2 = [float(v) for v in d.geom_xmat[g, 6:9]]
        c = [float(v) for v in m.geom_aabb[g, :3]]
        h = [float(v) for v in m.geom_aabb[g, 3:]]
        zc = float(d.geom_xpos[g, 2]) + R2[0] * c[0] + R2[1] * c[1] + R2[2] * c[2]
        zh = abs(R2[0]) * h[0] + abs(R2[1]) * h[1] + abs(R2[2]) * h[2]
        up = 0
        for j in (1, 2):
            if abs(R2[j]) > abs(R2[up]):
                up = j
        f = dict(top=zc + zh, height=2 * zh, thin=min(h[j] for j in range(3) if j != up),
                 moving=int(m.body_weldid[b]) != 0 or int(m.body_mocapid[b]) >= 0)
        cls = PALETTE["fallback"]
        for rule, nre, bre in _RULES:
            if nre is not None and not nre.search(gnames[g]):
                continue
            if bre is not None and not bre.search(bnames[b]):
                continue
            if "type" in rule and gtype not in rule["type"]:
                continue
            if "group" in rule and int(m.geom_group[g]) not in rule["group"]:
                continue
            if "moving" in rule and rule["moving"] != f["moving"]:
                continue
            if "top_max" in rule and not f["top"] <= rule["top_max"]:
                continue
            if "height_min" in rule and not f["height"] >= rule["height_min"]:
                continue
            if "height_max" in rule and not f["height"] <= rule["height_max"]:
                continue
            if "thin_max" in rule and not f["thin"] <= rule["thin_max"]:
                continue
            if "wide_min" in rule and not f["thin"] >= rule["wide_min"]:
                continue
            cls = rule["class"]
            break
        out.append(cls)
    return out


def cheap_geom_classes(m: mujoco.MjModel, own_root: int = -1) -> np.ndarray:
    """Each geom's class id [ngeom] uint8 (palette ids; 0 for a geom the camera never sees), as RaySight.cheapGeomClasses."""
    ids = PALETTE["ids"]
    return np.array([0 if c is None else ids[c] for c in geom_classes(m, own_root)], np.uint8)


def cheap_geom_colours(m: mujoco.MjModel, own_root: int = -1) -> np.ndarray:
    """Each geom's base colour in the cheap look [ngeom, 3] float32 (rayeye.ts #cheapLook + camera-look.ts
    cheapGeomLook): its class's palette colour; a geom the camera never sees keeps the head camera's colour rule (the
    geom's rgba, its material's when the rgba is MuJoCo's default grey). own_root: the camera's own duck's root body,
    whose geoms (self / self_accent) take the palette's dark neutral, not another duck's colours."""
    rgba, mid = m.geom_rgba.astype(np.float32), m.geom_matid
    out = rgba[:, :3].copy()
    dflt = (mid >= 0) & np.all(rgba == np.array([0.5, 0.5, 0.5, 1.0], np.float32), axis=1)
    out[dflt] = m.mat_rgba.astype(np.float32)[mid[dflt], :3]
    for g, c in enumerate(geom_classes(m, own_root)):
        if c is not None:
            out[g] = np.array(_class_colour(c), np.float32)
    return out


def _quat_mat(q: np.ndarray) -> np.ndarray:
    """camera.ts quatMat's formula (1 - 2(y^2 + z^2), ...), not mju_quat2Mat's (w^2 + x^2 - y^2 - z^2, ...): the two differ
    for CAM_QUAT, which is unit only to 6 digits, by ~3e-7, and that moved ray distances by ~1e-6 m."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def shade(m: mujoco.MjModel, d: mujoco.MjData, head: tuple, pos: np.ndarray, rot: np.ndarray, V: np.ndarray, t: np.ndarray,
          nrm: np.ndarray, base: np.ndarray) -> np.ndarray:
    """The head camera's shading (camera.ts render / shadeHit, warp's): rays V [n, 3] from pos hitting at distances t
    with normals nrm, surface colours base [n, 3] -> RGB [n, 3] clamped to 0..1 (misses: whatever; the caller puts the
    sky there). head = (active, ambient, diffuse, specular) of the model's headlight; rot the camera's rotation."""
    n = len(V)
    hp = pos[None, :] + V * t[:, None]
    vd = -V
    amb = np.zeros((n, 3))
    diff = np.zeros((n, 3))
    spec = np.zeros((n, 3))
    active, h_amb, h_diff, h_spec = head
    if active:
        amb += h_amb
    elif m.nlight == 0:
        amb += NO_LIGHT_AMBIENT

    def light(L: np.ndarray, att, ldiff: np.ndarray, lspec: np.ndarray) -> None:
        ndotl = np.maximum(0.0, (nrm * L).sum(1))
        on = ndotl > 0
        diff[on] += (ldiff[None, :] * (ndotl * att)[:, None])[on]
        H = L + vd
        H = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-300)
        ndoth = np.maximum(0.0, (nrm * H).sum(1))
        spec[on] += (lspec[None, :] * (MAT_SPECULAR * ndoth ** MAT_SHININESS * att)[:, None])[on]

    for l in range(m.nlight):
        if not m.light_active[l]:
            continue
        amb += m.light_ambient[l]
    for l in range(m.nlight):
        if not m.light_active[l]:
            continue
        if int(m.light_type[l]) == 1:
            L = np.broadcast_to(-d.light_xdir[l], (n, 3))
            att = np.ones(n)
        else:
            dv = d.light_xpos[l][None, :] - hp
            dist = np.linalg.norm(dv, axis=1)
            dist[dist == 0] = 1.0
            L = dv / dist[:, None]
            a = m.light_attenuation[l]
            q = a[0] + a[1] * dist + a[2] * dist * dist
            att = np.where(q != 0, 1.0 / np.where(q != 0, q, 1.0), 0.0)
            if int(m.light_type[l]) == 0:
                cos_t = -(L * d.light_xdir[l][None, :]).sum(1)
                att = np.where(cos_t < np.cos(np.radians(m.light_cutoff[l])), 0.0, att * np.maximum(cos_t, 0.0) ** m.light_exponent[l])
        light(L, att, m.light_diffuse[l], m.light_specular[l])
    if active:
        fwd = -rot[:, 2]
        light(np.broadcast_to(-fwd, (n, 3)), np.ones(n), h_diff, h_spec)  # the headlight: L = -lightdir, lightdir = forward
    return np.clip(base * (amb + diff) + spec, 0.0, 1.0)


@njit(cache=True)
def _cheap_kernel(V, t, nrm, g, rgb, pos, fwd, h_on, h_amb, h_diff, h_spec, nlight, l_on, l_type, l_amb, l_diff, l_spec,
                  l_att, l_cut, l_exp, l_pos, l_dir, out):
    """The cheap picture's rays as rayeye.ts does them, one by one in shadeHit's order of operations (camera.ts): depth
    into out[0], the shaded colour (or the sky) into out[1:4]; out is float32 [4, n]. RaySight passes zero specular (the
    palette look has no highlight), so s0..s2 stay 0, and the shading is capped at 1 (shadeHit's capShading)."""
    n = V.shape[0]
    for i in range(n):
        gi = g[i]
        if gi < 0:
            out[0, i] = 1.0
            out[1, i], out[2, i], out[3, i] = SKY[0], SKY[1], SKY[2]
            continue
        ti = t[i]
        out[0, i] = min(ti, RAY_DMAX) / RAY_DMAX
        v0, v1, v2 = V[i, 0], V[i, 1], V[i, 2]
        nx, ny, nz = nrm[i, 0], nrm[i, 1], nrm[i, 2]
        hp0, hp1, hp2 = pos[0] + v0 * ti, pos[1] + v1 * ti, pos[2] + v2 * ti
        vd0, vd1, vd2 = -v0, -v1, -v2
        a0 = a1 = a2 = 0.0
        s0 = s1 = s2 = 0.0
        if h_on:
            a0 += h_amb[0]
            a1 += h_amb[1]
            a2 += h_amb[2]
        elif nlight == 0:
            a0 += NO_LIGHT_AMBIENT
            a1 += NO_LIGHT_AMBIENT
            a2 += NO_LIGHT_AMBIENT
        for l in range(nlight):
            if l_on[l]:
                a0 += l_amb[l, 0]
                a1 += l_amb[l, 1]
                a2 += l_amb[l, 2]
        for l in range(nlight + (1 if h_on else 0)):
            if l < nlight:
                if not l_on[l]:
                    continue
                att = 1.0
                if l_type[l] == 1:
                    L0, L1, L2 = -l_dir[l, 0], -l_dir[l, 1], -l_dir[l, 2]
                else:
                    dx, dy, dz = l_pos[l, 0] - hp0, l_pos[l, 1] - hp1, l_pos[l, 2] - hp2
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if dist == 0.0:
                        dist = 1.0
                    L0, L1, L2 = dx / dist, dy / dist, dz / dist
                    q = l_att[l, 0] + l_att[l, 1] * dist + l_att[l, 2] * dist * dist
                    att = 1.0 / q if q != 0.0 else 0.0
                    if l_type[l] == 0:
                        cos_t = -(L0 * l_dir[l, 0] + L1 * l_dir[l, 1] + L2 * l_dir[l, 2])
                        att = 0.0 if cos_t < math.cos(l_cut[l] * math.pi / 180.0) else att * max(cos_t, 0.0) ** l_exp[l]
                d0, d1, d2 = l_diff[l, 0], l_diff[l, 1], l_diff[l, 2]
                p0, p1, p2 = l_spec[l, 0], l_spec[l, 1], l_spec[l, 2]
            else:  # the headlight: L = -forward
                att = 1.0
                L0, L1, L2 = -fwd[0], -fwd[1], -fwd[2]
                d0, d1, d2 = h_diff[0], h_diff[1], h_diff[2]
                p0, p1, p2 = h_spec[0], h_spec[1], h_spec[2]
            ndotl = max(0.0, nx * L0 + ny * L1 + nz * L2)
            if ndotl == 0.0:
                continue
            a0 += d0 * ndotl * att
            a1 += d1 * ndotl * att
            a2 += d2 * ndotl * att
            h0, h1, h2 = L0 + vd0, L1 + vd1, L2 + vd2
            hl = math.sqrt(h0 * h0 + h1 * h1 + h2 * h2)
            if hl == 0.0:
                hl = 1.0
            sp = MAT_SPECULAR * max(0.0, (nx * h0 + ny * h1 + nz * h2) / hl) ** MAT_SHININESS * att
            s0 += p0 * sp
            s1 += p1 * sp
            s2 += p2 * sp
        a0, a1, a2 = min(a0, 1.0), min(a1, 1.0), min(a2, 1.0)  # the palette look caps the shading at 1 (no clipping)
        out[1, i] = min(1.0, max(0.0, rgb[gi, 0] * a0 + s0))
        out[2, i] = min(1.0, max(0.0, rgb[gi, 1] * a1 + s1))
        out[3, i] = min(1.0, max(0.0, rgb[gi, 2] * a2 + s2))


class HeadCamera:
    """A duck's head camera on `model` (the duck whose names carry `prefix`), rendering RAY_RES x RAY_RES."""

    def __init__(self, model: mujoco.MjModel, prefix: str = "", res: int = RAY_RES):
        self.m, self.res = model, res
        self.cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, prefix + "head_camera")
        self.body = int(model.cam_bodyid[self.cam])
        self.local = _quat_mat(CAM_QUAT)
        half = np.tan(np.radians(FOVY) / 2)
        c = (np.arange(res) + 0.5) / res
        x = -half + 2 * half * c[None, :].repeat(res, 0)
        y = (half - 2 * half * c)[:, None].repeat(res, 1)
        d = np.stack([x, y, -np.ones_like(x)], -1).reshape(-1, 3)
        self.dirs = d / np.linalg.norm(d, axis=1, keepdims=True)
        n = res * res
        self.geomid, self.dist, self.normal = np.zeros(n, np.int32), np.zeros(n), np.zeros(3 * n)
        hl = model.vis.headlight
        self.head = (bool(hl.active), np.array(hl.ambient, float), np.array(hl.diffuse, float), np.array(hl.specular, float))
        self.rgba = model.geom_rgba.astype(np.float64)
        self.mat = model.geom_matid.copy()
        self.mat_rgba = model.mat_rgba.astype(np.float64)

    def pose(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """The camera's world position and rotation [3, 3] (columns = its x right, y up, z back axes), camera.ts pose."""
        b = self.body
        R = d.xmat[b].reshape(3, 3)
        return d.xpos[b] + R @ self.m.cam_pos[self.cam], R @ self.local

    def render(self, d: mujoco.MjData) -> np.ndarray:
        """The view at `d` (forwarded): [res, res, 3] RGB in 0..1, rows from the top."""
        m, b = self.m, self.body
        pos, rot = self.pose(d)
        v = (self.dirs @ rot.T).reshape(-1)
        n = self.res * self.res
        mujoco.mj_multiRay(m, d, pos, v, GROUPS, True, b, self.geomid, self.dist, self.normal, n, CUTOFF)
        g, t, nrm, V = self.geomid, self.dist, self.normal.reshape(n, 3), v.reshape(n, 3)
        hit = g >= 0
        base = self.rgba[np.maximum(g, 0), :3].copy()
        mid = self.mat[np.maximum(g, 0)]
        dflt = (mid >= 0) & np.all(self.rgba[np.maximum(g, 0)] == np.array([0.5, 0.5, 0.5, 1.0]), axis=1)
        base[dflt] = self.mat_rgba[mid[dflt], :3]
        out = shade(m, d, self.head, pos, rot, V, t, nrm, base)
        out[~hit] = SKY
        return out.reshape(self.res, self.res, 3)


class RaySight:
    """One duck's camera rays (rayeye.ts RaySight): refresh(d, cheap, class_ids) writes the palette picture.

    RAY_COLS x RAY_ROWS of the camera's 64 x 64 pixel centres (every second column, every fourth row), one mj_multiRay
    at the palette picture's groups (PICTURE_GROUPS) leaving out the camera's body."""

    def __init__(self, model: mujoco.MjModel, prefix: str = ""):
        self.m = model
        self.camera = HeadCamera(model, prefix, res=1)  # its pose only
        t = math.tan(0.5 * FOVY * math.pi / 180)
        dirs = []
        for py in range(RAY_ROW_STRIDE >> 1, RAY_RES, RAY_ROW_STRIDE):
            for px in range(RAY_COL_STRIDE >> 1, RAY_RES, RAY_COL_STRIDE):
                x = -t + (2 * t * (px + 0.5)) / RAY_RES
                y = t - (2 * t * (py + 0.5)) / RAY_RES
                l = math.hypot(x, y, 1)
                dirs.append((x / l, y / l, -1 / l))
        self.dirs = np.array(dirs)
        n = len(dirs)
        # the palette picture's hits (PICTURE_GROUPS)
        self.pic_geomid, self.pic_dist, self.pic_normal = np.zeros(n, np.int32), np.zeros(n), np.zeros(3 * n)
        self.own_root = int(model.body_rootid[self.camera.body])
        self._cheap_rgb: np.ndarray | None = None  # the palette look's geom colours, made on the first picture

    def cheap_geom_classes(self) -> np.ndarray:
        """Each geom's class id in the palette look for this duck's camera (its own duck: self / self_accent)."""
        if getattr(self, "_cheap_ids", None) is None:
            self._cheap_ids = cheap_geom_classes(self.m, self.own_root)
        return self._cheap_ids

    def refresh(self, d: mujoco.MjData, cheap: np.ndarray | None = None, class_ids: np.ndarray | None = None) -> None:
        """At `d` (forwarded), with `cheap` (float32 [4, RAY_ROWS, RAY_COLS], PICTURE_SHAPE) the rays' picture in the
        shared palette ("ray-rgbd-palette-4x16x32/2", rayeye.ts refresh): channel 0 depth min(dist, RAY_DMAX) / RAY_DMAX
        (1 = no hit), channels 1-3 the hit's colour (cheap_geom_colours: its class's palette colour) times the head
        camera's ambient + diffuse shading capped at 1 (no specular highlight), the sky where nothing is hit. With
        `class_ids` (uint8 [RAY_ROWS, RAY_COLS]) also each ray's class id (palette ids, 0 = nothing hit), so a picture can
        be stored as classes and repainted in another palette. One cast at PICTURE_GROUPS: the physics geoms, the ducks'
        camera proxies and the Challenges' visual-only parts."""
        if cheap is None and class_ids is None:
            return
        m, cam_body = self.m, self.camera.body
        pos, rot = self.camera.pose(d)
        V = self.dirs @ rot.T
        n = len(V)
        mujoco.mj_multiRay(m, d, pos, V.reshape(-1), PICTURE_GROUPS, True, cam_body, self.pic_geomid, self.pic_dist,
                           self.pic_normal, n, CUTOFF)
        if cheap is not None:
            if self._cheap_rgb is None:
                self._cheap_rgb = cheap_geom_colours(m, self.own_root).astype(np.float64)
                hl = m.vis.headlight
                f64 = lambda x: np.ascontiguousarray(x, np.float64)  # noqa: E731
                # no specular highlight in the palette look (rayeye.ts refresh, docs/vision/PALETTE.md): the lights' and
                # the headlight's specular are zero, so a hit is its class colour * min(1, the shading)
                self._look = (bool(hl.active), f64(hl.ambient), f64(hl.diffuse), np.zeros(3), int(m.nlight),
                              np.ascontiguousarray(m.light_active, np.bool_), np.ascontiguousarray(m.light_type, np.int64),
                              f64(m.light_ambient).reshape(-1, 3), f64(m.light_diffuse).reshape(-1, 3),
                              np.zeros((int(m.nlight), 3)), f64(m.light_attenuation).reshape(-1, 3),
                              f64(m.light_cutoff), f64(m.light_exponent))
            _cheap_kernel(np.ascontiguousarray(V), self.pic_dist, self.pic_normal.reshape(n, 3), self.pic_geomid, self._cheap_rgb, pos,
                          np.ascontiguousarray(-rot[:, 2]), *self._look, d.light_xpos.reshape(-1, 3),
                          d.light_xdir.reshape(-1, 3), cheap.reshape(4, n))
        if class_ids is not None:
            pg = self.pic_geomid
            class_ids.reshape(-1)[:] = np.where(pg >= 0, self.cheap_geom_classes()[np.maximum(pg, 0)], 0)
