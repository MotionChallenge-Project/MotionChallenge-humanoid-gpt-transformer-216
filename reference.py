"""Author EMA and reference features, independent of scoring and simulation."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from motion_challenge.adapters.onnx import yaw
from motion_challenge.validation import sha256_file


def quat_heading(quaternion):
    """Pinned author's NumPy formula, preserving float32 reference rounding."""
    w, x, y, z = np.asarray(quaternion)[None].T
    angle = np.arctan2(2. * (x*y + w*z), 1. - 2. * (y*y + z*z))
    return np.arctan2(np.sin(angle), np.cos(angle))[0]


class ReferenceFeatures:
    def __init__(self, root, runtime, names):
        path = root / runtime['kinematics']
        if sha256_file(path) != runtime['kinematics_sha256']:
            raise ValueError('kinematics checksum mismatch')
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.qadr = [self.model.joint(n).qposadr[0] for n in names]
        self.vadr = [self.model.joint(n).dofadr[0] for n in names]
        self.pelvis = self.model.body('pelvis').id
        self.dt = runtime['control_period_s']

    def reset(self, order):
        self.order = order
        self.previous = None
        self.cache = {}

    def update(self, frames):
        features = [self._frame(frame) for frame in frames]
        keep = {round(frame.source_time_s, 9) for frame in frames}
        self.cache = {key: value for key, value in self.cache.items() if key in keep}
        return features

    def _frame(self, frame):
        key = round(frame.source_time_s, 9)
        if key in self.cache:
            return self.cache[key]
        raw = np.r_[frame.root_position_world_m,
                    np.asarray(frame.root_orientation_xyzw)[[3, 0, 1, 2]],
                    np.asarray(frame.joint_position_rad)[self.order]].astype(np.float32)
        if self.previous is None:
            smoothed = raw.copy()
        else:
            previous = self.previous
            quaternion = raw[3:7]
            if np.dot(previous[3:7], quaternion) < 0:
                quaternion = -quaternion
            smoothed = .8 * previous + .2 * raw
            blended = .8 * previous[3:7] + .2 * quaternion
            smoothed[3:7] = blended / max(np.linalg.norm(blended), 1e-8)
        # Preserve float32 EMA state. The author's following resampler
        # normalizes quaternion values in float64.
        normalized = smoothed.astype(np.float64)
        rotation = R.from_quat(normalized[3:7], scalar_first=True)
        normalized[3:7] = rotation.as_quat(scalar_first=True)
        velocity = np.zeros(35)
        if self.previous is not None:
            old = R.from_quat(self.previous[3:7], scalar_first=True)
            velocity[:3] = (normalized[:3] - self.previous[:3]) / self.dt
            # Retain the author's world-frame qvel convention before FK.
            velocity[3:6] = old.apply((old.inv() * rotation).as_rotvec()) / self.dt
            velocity[6:] = (normalized[7:] - self.previous[7:]) / self.dt
        self.data.qpos[:7] = normalized[:7].astype(np.float32)
        self.data.qpos[self.qadr] = normalized[7:].astype(np.float32)
        self.data.qvel[:6] = velocity[:6].astype(np.float32)
        self.data.qvel[self.vadr] = velocity[6:].astype(np.float32)
        mujoco.mj_forward(self.model, self.data)
        cvel = self.data.cvel[self.pelvis].astype(np.float32)
        heading = R.from_euler('z', float(yaw(rotation))).as_matrix().astype(np.float32)
        value = {'qpos': normalized.astype(np.float32),
                 'gravity': -self.data.xmat[self.pelvis].reshape(3, 3)[2].astype(np.float32),
                 'cvel': np.r_[heading.T @ cvel[:3], heading.T @ cvel[3:]]}
        self.previous = smoothed
        self.cache[key] = value
        return value
