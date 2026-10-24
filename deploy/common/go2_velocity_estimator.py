"""Flat-ground kinematic velocity baseline from joint encoders and IMU only.

No base position/velocity, simulator contact state or contact force is accepted.
Stationary stance feet are assumed; sustained slip is not observable here.
"""
import mujoco
import numpy as np


class KinematicVelocityEstimator:
    def __init__(self, xml_path, sdk_joint_names, filter_seconds=.08, height_scale=.01):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.qpos_ids = np.array([self.model.joint(n).qposadr[0] for n in sdk_joint_names])
        self.qvel_ids = np.array([self.model.joint(n).dofadr[0] for n in sdk_joint_names])
        self.base = self.model.body("base_link").id
        self.feet = []
        for leg in ("FR", "FL", "RR", "RL"):
            body = self.model.body(f"{leg}_calf").id
            matches = np.flatnonzero((self.model.geom_bodyid == body)
                & (self.model.geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE))
                & np.isclose(self.model.geom_size[:, 0], .022))
            if len(matches) != 1:
                raise ValueError(f"Cannot identify unique foot sphere for {leg}")
            self.feet.append(int(matches[0]))
        self.filter_seconds = float(filter_seconds)
        self.height_scale = float(height_scale)
        if (self.filter_seconds < 0 or not np.isfinite(self.filter_seconds)
                or not np.isfinite(self.height_scale) or self.height_scale <= 0):
            raise ValueError("Invalid estimator filter/height scale")
        self.filtered = None
        self.weights = np.zeros(4)

    def estimate(self, q_sdk, dq_sdk, quaternion_wxyz, gyro_body, dt):
        q, dq, quat, gyro = map(lambda a: np.asarray(a, dtype=float),
                               (q_sdk, dq_sdk, quaternion_wxyz, gyro_body))
        if (q.shape != (12,) or dq.shape != (12,) or quat.shape != (4,) or gyro.shape != (3,)
                or not all(np.isfinite(v).all() for v in (q, dq, quat, gyro))
                or not np.isfinite(dt) or dt <= 0 or np.linalg.norm(quat) < 1e-8):
            raise ValueError("Invalid kinematic estimator input")
        self.data.qpos[:3] = [0, 0, .5]  # Arbitrary translation, cancels in relative FK.
        self.data.qpos[3:7] = quat / np.linalg.norm(quat)
        self.data.qpos[self.qpos_ids] = q
        self.data.qvel[:] = 0  # Explicitly do not use simulator base velocity.
        mujoco.mj_forward(self.model, self.data)
        rotation = self.data.xmat[self.base].reshape(3, 3)
        candidates, heights = [], []
        for geom in self.feet:
            # No-slip applies to the material point touching the floor, not
            # the center of a rotating spherical foot.
            point = self.data.geom_xpos[geom].copy()
            point[2] -= self.model.geom_size[geom, 0]
            relative_world = point - self.data.xpos[self.base]
            relative_body = rotation.T @ relative_world
            jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jacp, jacr, point, int(self.model.geom_bodyid[geom]))
            joint_velocity = rotation.T @ jacp[:, self.qvel_ids] @ dq
            candidates.append(-np.cross(gyro, relative_body) - joint_velocity)
            heights.append(self.data.geom_xpos[geom, 2] - self.data.xpos[self.base, 2])
        heights = np.asarray(heights)
        weights = np.exp(-(heights - heights.min()) / self.height_scale)
        self.weights = weights / weights.sum()
        raw = self.weights @ np.asarray(candidates)
        alpha = dt / (self.filter_seconds + dt)
        self.filtered = raw if self.filtered is None else self.filtered + alpha * (raw - self.filtered)
        return self.filtered.copy()
