"""Plant v0 reference loop: docs/plant-v0.md §3-5, nothing else.

Loads ONLY the built bundle packages/sim/plant/microduck-legs-pd-v0/ (plant.xml, manifest.json,
policies/*.onnx). Citations use two roots:
  MJLAB = data/upstream/pypi__mjlab-1.3.0/mjlab                  (training framework, 1.3.0)
  MDRL  = data/upstream/pollen-robotics__microduck_rl/src/mjlab_microduck   (microduck_rl @ cb70b79)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

# The repo (tools/py/duckref/ -> 3 up). A copy outside it (the players' RL kit) gets its plant from DUCKBENCH_PLANT_DIR
# (bam.py) and may sit too close to / for 3 parents.
_parents = Path(__file__).resolve().parents
REPO = _parents[3] if len(_parents) > 3 else _parents[-1]
BUNDLE_DIR = REPO / "packages" / "sim" / "plant" / "microduck-legs-pd-v0"

HEAD_EMA = 0.2  # plant-v0 §5.2: h <- h + 0.2 * (h_target - h)
BASE_Z = 0.12  # plant-v0 §4: STAND keyframe base height
GRAVITY_DIR_W = np.array([0.0, 0.0, -1.0])  # MJLAB/entity/entity.py:787 gravity_vec_w


def phase_block(phi: float) -> np.ndarray:
    """Command block [48:51] of phase skills: [cos 2*pi*phi, sin 2*pi*phi, 0]."""
    return np.array([np.cos(2.0 * np.pi * phi), np.sin(2.0 * np.pi * phi), 0.0])


def sitstand_block(flag: float) -> np.ndarray:
    """Command block [48:51] of alpha_sitstand: [flag, 0, 0] (1 = sit, 0 = stand)."""
    return np.array([float(flag), 0.0, 0.0])


def load_manifest(bundle_dir: str | Path = BUNDLE_DIR) -> dict:
    return json.loads((Path(bundle_dir) / "manifest.json").read_text())


# The duck's camera proxies (packages/schema/src/palette.ts DUCK_PROXIES, also in palette.json "duck_proxies"; the kit's
# tools/vision_test.py checks they are equal): one box per duck body without a collision shape, plus the trunk's shell.
DUCK_PROXIES = [
    {"body": "trunk_base", "pos": [-0.0062, -0.0002, -0.001], "size": [0.0407, 0.032, 0.043]},
    {"body": "yaw2roll", "pos": [0, 0.001, 0.017], "size": [0.0115, 0.0155, 0.02]},
    {"body": "bearing_roll", "pos": [0, 0.001, 0.017], "size": [0.0115, 0.0155, 0.02]},
    {"body": "upper_leg_left", "pos": [0.0104, 0.0183, 0.012], "size": [0.0238, 0.0305, 0.016]},
    {"body": "upper_leg_right", "pos": [-0.0104, 0.0183, 0.012], "size": [0.0238, 0.0305, 0.016]},
    {"body": "neck", "pos": [0, -0.025, 0.0145], "size": [0.01, 0.0345, 0.0145]},
    {"body": "neck_pitch", "pos": [0, 0.0059, -0.0145], "size": [0.011, 0.0149, 0.0175]},
    {"body": "yaw_roll_motion", "pos": [-0.001, -0.0068, 0.0145], "size": [0.019, 0.0178, 0.0145]},
]
CAMERA_PROXY_GROUP = 4
# packages/sim/src/plant.ts MATCH_DUCK_PREFIXES
MATCH_DUCK_PREFIXES = ("", "b_", "c_", "d_", "e_", "f_", "g_", "h_")


CAMERA_GEOMS_MARK = "<!-- duck-arena camera-geoms -->"  # plant.ts CAMERA_GEOMS_MARK


def with_camera_geoms(xml: str) -> str:
    """plant.ts withCameraGeoms: the world a run with a palette camera move compiles (physics_only then adds camera_geoms);
    blind and ray-eye runs compile the plain XML, main's model."""
    if CAMERA_GEOMS_MARK in xml:
        return xml
    out, n = re.subn(r"<compiler\b", CAMERA_GEOMS_MARK + "<compiler", xml, count=1)
    if not n:
        raise ValueError("MJCF has no <compiler> element")
    return out


_TOK = re.compile(r"<body\b[^>]*>|</body>|<(?:joint|freejoint)\b[^>]*>|<geom\b[^>]*/>")


def _moving_body_geoms(xml: str) -> list:
    """packages/sim/src/plant.ts movingBodyGeoms: per <geom .../>, in order, whether its body moves freely (a free joint on it or an ancestor; R16-1)."""
    has_joint, collides, open_ = [], [], []
    for m in _TOK.finditer(xml):
        t = m[0]
        if t.startswith("<body"):
            if not t.endswith("/>"):
                open_.append(len(has_joint)); has_joint.append(False); collides.append(False)
        elif t == "</body>":
            open_.pop()
        elif t.startswith("<geom"):  # the body has a collision geom of its own
            if open_ and not (re.search(r'\bcontype="0"', t) and re.search(r'\bconaffinity="0"', t)):
                collides[open_[-1]] = True
        elif open_ and (t.startswith("<freejoint") or (t.startswith("<joint") and re.search(r'\btype="free"', t))):
            has_joint[open_[-1]] = True  # free joints only (R16-1): hinge and slide bodies keep their visual parts
    stack, out, b = [], [], 0
    for m in _TOK.finditer(xml):
        t = m[0]
        if t.startswith("<body"):
            if not t.endswith("/>"):
                stack.append((has_joint[b] or (bool(stack) and stack[-1][0]), collides[b])); b += 1
        elif t == "</body>":
            stack.pop()
        elif t.startswith("<geom"):
            out.append(bool(stack) and stack[-1][0] and stack[-1][1])  # only beside a collision geom on a free body
    return out


def camera_geoms(xml: str) -> str:
    """packages/sim/src/plant.ts cameraGeoms: what the palette picture sees beyond physics, added to every sim model. The
    Challenges' visual-only parts in group 5 (contype = conaffinity = 0) get conaffinity 8 so discardvisual keeps them,
    and each duck body in DUCK_PROXIES (any match prefix) gets its camera proxy box as its first geom (group 4, contype
    0, conaffinity 8, mass 0: nothing collides with them, and the bodies' own inertials keep the dynamics)."""
    moving = iter(_moving_body_geoms(xml))  # Emil's PR #1 R15-1 / R16-1: no visual part kept on a free-moving body

    def keep(m: re.Match) -> str:
        g = m[0]
        if (not next(moving) and re.search(r'\bgroup="5"', g) and re.search(r'\bcontype="0"', g)
                and re.search(r'\bconaffinity="0"', g)):
            return re.sub(r'\bconaffinity="0"', 'conaffinity="8"', g, count=1)
        return g
    kept = re.sub(r"<geom\b[^>]*/>", keep, xml)
    by_body = {p["body"]: p for p in DUCK_PROXIES}

    def num(x) -> str:
        return json.dumps(x)

    def proxy(m: re.Match) -> str:
        tag, name = m[0], m[1]
        if tag.endswith("/>"):
            return tag
        for pre in MATCH_DUCK_PREFIXES:
            if name.startswith(pre) and name[len(pre):] in by_body:
                p = by_body[name[len(pre):]]
                return (tag + f'<geom name="{pre}{p["body"]}_camproxy" type="box" group="{CAMERA_PROXY_GROUP}" contype="0" '
                        f'conaffinity="8" mass="0" pos="{" ".join(map(num, p["pos"]))}" size="{" ".join(map(num, p["size"]))}"/>')
        return tag
    return re.sub(r'<body\b[^>]*\bname="([^"]*)"[^>]*>', proxy, kept)


def physics_only(xml: str) -> str:
    """packages/sim/src/plant.ts physicsOnly, as the TS sim compiles every MJCF: MuJoCo's discardvisual (geoms with
    contype = conaffinity = 0, the duck's visual meshes and Challenge markers, are dropped; the same dynamics), and the text
    loses the visual mesh geoms (class "visual" without contype/conaffinity of their own) and the mesh files nothing
    else uses, so the visual STLs are never read (plant v1's collision meshes are inline hulls: no mesh file left).
    Only when every body declares its <inertial> (MuJoCo computes inertia from all geoms before it discards any). Plus
    camera_geoms in a world marked by with_camera_geoms only: the palette picture's proxies and visual-only parts."""
    out, n = re.subn(r"<compiler\b", '<compiler discardvisual="true"', xml, count=1)
    if not n:
        raise ValueError("MJCF has no <compiler> element")
    if CAMERA_GEOMS_MARK in out:  # only a world marked by with_camera_geoms (a palette camera move's run)
        out = camera_geoms(out)
    own_inertia = not re.search(r'\binertiafromgeom="true"', xml) and \
        len(re.findall(r"<body\b", xml)) == len(re.findall(r"<inertial\b", xml))
    if not own_inertia:
        return out
    kept = re.sub(r'<geom\b[^>]*\bclass="visual"[^>]*/>',
                  lambda m: m[0] if re.search(r"\b(contype|conaffinity)=", m[0]) else "", out)
    used = set(re.findall(r'<geom\b[^>]*\bmesh="([^"]+)"', kept))

    def mesh(m: re.Match) -> str:
        name = re.search(r'\bname="([^"]+)"', m[0])
        return m[0] if (name[1] if name else re.sub(r"\.[^.]*$", "", m[1].rsplit("/", 1)[-1])) in used else ""
    return re.sub(r'<mesh\b[^>]*\bfile="([^"]+)"[^>]*/>', mesh, kept)


def load_model(mjcf: str | Path) -> mujoco.MjModel:
    """A bundle's MJCF as MuJoCo compiles it; a bundle without meshes/ (the players' RL kit gets the plant without its
    visual STLs, docs/player-kit-decisions-log.md PK-07) is compiled physics only, the same dynamics."""
    mjcf = Path(mjcf)
    if (mjcf.parent / "meshes").is_dir():
        return mujoco.MjModel.from_xml_path(str(mjcf))
    return mujoco.MjModel.from_xml_string(physics_only(mjcf.read_text()))


class DuckRef:
    """One duck on plant v0. `reset()` then `tick(...)` every 0.02 s."""

    def __init__(self, bundle_dir: str | Path = BUNDLE_DIR, model: mujoco.MjModel | None = None):
        self.bundle_dir = Path(bundle_dir)
        self.manifest = mf = load_manifest(self.bundle_dir)
        mjcf = next(f["path"] for f in mf["files"] if f["role"] == "mjcf")
        self.model = m = model if model is not None else load_model(self.bundle_dir / mjcf)
        self.data = mujoco.MjData(m)
        assert m.opt.timestep == mf["timestep"], (m.opt.timestep, mf["timestep"])

        self.joint_names: list[str] = list(mf["joint_names"])  # plant-v0 §3, ONNX metadata order
        self.default_pos = np.array(mf["default_pos"], dtype=np.float64)  # MDRL/robot/microduck_constants.py:116-140
        self.decimation = int(mf["decimation"])  # 4 -> 50 Hz policy

        jids = [m.joint(n).id for n in self.joint_names]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jids])
        self.vadr = np.array([m.jnt_dofadr[j] for j in jids])
        act_of_joint = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)
                        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
        self.act = np.array([act_of_joint[j] for j in jids])
        self.trunk = m.body("trunk_base").id
        self.gyro_adr = int(m.sensor_adr[m.sensor("imu_ang_vel").id])
        self.stand_key = m.key("STAND").id

        self._sessions: dict[str, ort.InferenceSession] = {}
        self.reset()

    # -- policies -----------------------------------------------------------------------------

    def session(self, policy: str) -> ort.InferenceSession:
        if policy not in self._sessions:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            path = self.bundle_dir / self.manifest["policies"][policy]["onnx"]
            self._sessions[policy] = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        return self._sessions[policy]

    def command_names(self, policy: str) -> list[str]:
        return list(self.manifest["policies"][policy]["command_names"])

    # -- plant-v0 §4 ----------------------------------------------------------------------------

    def reset(self) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, self.stand_key)  # qpos = STAND, qvel = 0, time = 0
        assert d.qpos[2] == BASE_Z
        d.qpos[self.qadr] = self.default_pos  # servo joints = DEFAULT_POS (training HOME_FRAME)
        d.qvel[:] = 0.0
        d.ctrl[self.act] = self.default_pos
        self.prev_action = np.zeros(14, np.float32)  # MJLAB/managers/action_manager.py:146 (zeroed on reset)
        self.head_cmd = np.zeros(4)  # head_cmd_smoothed
        self.lag_joint_vel: np.ndarray | None = None  # empty; filled at tick 0
        self.k = 0

    # -- plant-v0 §5 ----------------------------------------------------------------------------

    def build_obs(self, policy: str, twist3, head_target4, body6) -> tuple[np.ndarray, dict]:
        """Step 2. Reads self.data right after mj_forward; returns (obs float64[61], parts)."""
        m, d = self.model, self.data
        names = self.command_names(policy)

        # [0:3] ang vel in the trunk(=IMU) frame. Training: MDRL/tasks/mdp.py:3794-3802 (root_link_ang_vel_b,
        # MJLAB/entity/data.py:600-602) replacing the gyro term MJLAB/tasks/velocity/velocity_env_cfg.py:82-86.
        ang_vel = d.sensordata[self.gyro_adr : self.gyro_adr + 3].copy()
        # [3:6] R(xquat)^T (0,0,-1). MJLAB/entity/data.py:584-586 (quat_apply_inverse(root_link_quat_w, g)).
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, d.xquat[self.trunk])
        grav = R.reshape(3, 3).T @ GRAVITY_DIR_W
        # [6:20] qpos - default. MJLAB/envs/mdp/observations.py:51-61 (joint_pos_rel; encoder bias DR not used).
        joint_pos = d.qpos[self.qadr] - self.default_pos
        # [20:34] qvel sampled at tick k-1 (tick 0: current). MJLAB/envs/mdp/observations.py:64-72 (joint_vel_rel)
        # delayed by MDRL/tasks/microduck_velocity_env_cfg.py:611-613 (lag 1) via
        # MJLAB/utils/buffers/delay_buffer.py:229-245 + circular_buffer.py:210-213 (first push back-fills).
        vel_now = d.qvel[self.vadr].copy()
        joint_vel = vel_now if self.lag_joint_vel is None else self.lag_joint_vel
        # [34:48] raw previous action. MJLAB/envs/mdp/observations.py:80-82 -> action_manager.py:114-117.
        prev_action = self.prev_action.astype(np.float64)
        # [48:51] twist / skill block. MJLAB/tasks/velocity/velocity_env_cfg.py:100-103 (generated_commands "twist").
        twist = np.asarray(twist3, dtype=np.float64)
        # [51:55] head, EMA-smoothed (runtime/Space convention, plant-v0 §5.2); zero unless trained with head_pose.
        # MDRL/tasks/microduck_velocity_env_cfg.py:694-697 (head_command term).
        self.head_cmd = self.head_cmd + HEAD_EMA * (np.asarray(head_target4, dtype=np.float64) - self.head_cmd)
        head = self.head_cmd if "head_pose" in names else np.zeros(4)
        # [55:61] body pose; zeros in v0 unless given, zero unless trained with body_pose.
        # MDRL/tasks/microduck_velocity_env_cfg.py:698-701 (body_command term).
        body = np.zeros(6) if body6 is None or "body_pose" not in names else np.asarray(body6, dtype=np.float64)

        # Term order: MJLAB/tasks/velocity/velocity_env_cfg.py:76-103 + MDRL/.../microduck_velocity_env_cfg.py:692-701
        obs = np.concatenate([ang_vel, grav, joint_pos, joint_vel, prev_action, twist, head, body])
        assert obs.shape == (61,)
        return obs, dict(vel_now=vel_now, joint_vel=joint_vel, prev_action=self.prev_action.copy(),
                         command13=obs[48:61].copy())

    def tick(self, policy: str, twist3=(0.0, 0.0, 0.0), head_target4=(0.0, 0.0, 0.0, 0.0), body6=None) -> dict:
        """One policy tick k (plant-v0 §5, steps 1-6). Returns the per-tick record."""
        m, d = self.model, self.data
        qpos, qvel = d.qpos.copy(), d.qvel.copy()
        qacc_warmstart = d.qacc_warmstart.copy()  # not part of the contract; lets a replay reproduce one step exactly

        # 1. Derived quantities (xquat, sensors) from the current qpos/qvel, as training does before obs
        #    (MJLAB/envs/manager_based_rl_env.py:447-457).
        mujoco.mj_forward(m, d)
        # 2. obs_k: float64, cast to float32 once.
        obs64, parts = self.build_obs(policy, twist3, head_target4, body6)
        obs = obs64.astype(np.float32)
        # 3. a_k = policy(obs_k).
        action = self.session(policy).run(None, {"obs": obs[None, :]})[0][0].astype(np.float32)
        # 4. ctrl = default + 1.0 * a (MJLAB/envs/mdp/actions/actions.py:154-157, 219-224;
        #    scale MDRL/tasks/microduck_velocity_env_cfg.py:273). No clip, no filter, no delay.
        ctrl = self.default_pos + 1.0 * action.astype(np.float64)
        d.ctrl[self.act] = ctrl
        # 5. Hold the targets for `decimation` physics steps (MJLAB/envs/manager_based_rl_env.py:414-420).
        for _ in range(self.decimation):
            mujoco.mj_step(m, d)
        # 6. Loop state for tick k+1. Neither is reset on a policy switch.
        self.prev_action = action
        self.lag_joint_vel = parts["vel_now"]

        rec = dict(k=self.k, policy=policy, qpos=qpos, qvel=qvel, lag_joint_vel=parts["joint_vel"],
                   prev_action=parts["prev_action"], command13=parts["command13"], obs=obs, action=action,
                   ctrl=ctrl, qacc_warmstart=qacc_warmstart)
        self.k += 1
        return rec

    # -- helpers ----------------------------------------------------------------------------------

    def tilt_deg(self) -> float:
        """Angle between the trunk z axis and world up, from the current xmat (call after mj_forward)."""
        return float(np.degrees(np.arccos(np.clip(self.data.xmat[self.trunk][8], -1.0, 1.0))))
