"""Simulation-only three-axis velocity feedback. Never imported by real control."""
import numpy as np


class VectorVelocityPI:
    def __init__(self, bounds, kp=(.8, .8, .8), ki=(.8, .8, .8),
                 filter_seconds=.2, slew=1.):
        self.bounds = np.asarray(bounds, dtype=float)
        self.kp, self.ki = np.broadcast_to(kp, (3,)).copy(), np.broadcast_to(ki, (3,)).copy()
        if (self.bounds.shape != (3, 2) or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[:, 0] >= 0) or np.any(self.bounds[:, 1] <= 0)
                or not np.isfinite([*self.kp, *self.ki, filter_seconds, slew]).all()
                or np.any(self.kp < 0) or np.any(self.ki < 0)
                or filter_seconds < 0 or slew <= 0):
            raise ValueError("Invalid velocity feedback parameters")
        self.filter_seconds, self.slew = filter_seconds, slew
        self.reference = np.zeros(3)
        self.command = np.zeros(3)
        self.integral = np.zeros(3)
        self.filtered = None
        self.saturated_steps = np.zeros(3, dtype=int)

    def update(self, target, measured, dt, active_axes=None):
        target, measured = np.asarray(target, dtype=float), np.asarray(measured, dtype=float)
        if (target.shape != (3,) or measured.shape != (3,)
                or not np.isfinite([*target, *measured, dt]).all() or dt <= 0):
            raise ValueError("Invalid velocity reference, feedback or time step")
        self.reference += np.clip(target - self.reference, -self.slew * dt, self.slew * dt)
        alpha = dt / (self.filter_seconds + dt)
        self.filtered = measured.copy() if self.filtered is None else self.filtered + alpha * (measured - self.filtered)
        error = self.reference - self.filtered
        if active_axes is not None:
            active = np.asarray(active_axes, dtype=bool)
            if active.shape != (3,):
                raise ValueError('Active-axis mask must have three components')
            error = np.where(active, error, 0.)
            self.integral = np.where(active, self.integral, 0.)
        candidate = self.integral + error * dt
        raw = self.reference + self.kp * error + self.ki * candidate
        lo, hi = self.bounds.T
        blocked = ((raw > hi) & (error > 0)) | ((raw < lo) & (error < 0))
        self.integral = np.where(blocked, self.integral, candidate)
        raw = self.reference + self.kp * error + self.ki * self.integral
        self.saturated_steps += (raw < lo) | (raw > hi)
        self.command += np.clip(np.clip(raw, lo, hi) - self.command, -self.slew * dt, self.slew * dt)
        return self.command.copy()


def stance_velocity_damping(model, data, joint_dofs, measured, reference, gain,
                            attitude_kp=0., attitude_kd=0., yaw_gain=None):
    """Oracle-contact simulation ablation: bounded stance Jacobian feedback.

    Return actuator torque opposing base velocity error. Requires MuJoCo
    contact and velocity truth; never a deployable state estimator.
    """
    import mujoco
    feet = {model.geom(name).id for name in ('FR', 'FL', 'RR', 'RL')}
    contacts = {}
    force = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        foot = c.geom1 if c.geom1 in feet else c.geom2 if c.geom2 in feet else None
        if foot is None:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        if force[0] > 1.:
            contacts[foot] = (c.pos.copy(), int(model.geom_bodyid[foot]))
    if not contacts:
        return np.zeros(len(joint_dofs))
    base = model.body('base_link').id
    rotation = data.xmat[base].reshape(3, 3)
    error = np.asarray(reference) - measured
    mass = float(model.body_mass.sum())
    world_force = rotation @ np.array([mass * gain * error[0], mass * gain * error[1], 0.])
    world_force = np.clip(world_force, -40., 40.)
    moment = float(np.clip(.5 * (gain if yaw_gain is None else yaw_gain) * error[2], -4., 4.))
    allocated = None
    if attitude_kp > 0 or attitude_kd > 0:
        w, x, y, z = data.qpos[3:7]
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1., 1.))
        gyro = data.sensor('imu_gyro').data
        body_moment = np.array([-attitude_kp * roll - attitude_kd * gyro[0],
                                -attitude_kp * pitch - attitude_kd * gyro[1], moment])
        world_moment = rotation @ np.clip(body_moment, -4., 4.)
        blocks = []
        for point, _ in contacts.values():
            x, y, z = point - data.xpos[base]
            skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
            blocks.append(np.vstack([np.eye(3), skew]))
        wrench_map = np.hstack(blocks)
        wrench = np.r_[world_force, world_moment]
        allocated = (wrench_map.T @ np.linalg.solve(wrench_map @ wrench_map.T + .002 * np.eye(6), wrench)).reshape(-1, 3)
        allocated = np.clip(allocated, -30., 30.)
    center = np.mean([p for p, _ in contacts.values()], axis=0)
    denom = sum(np.sum((p[:2] - center[:2]) ** 2) for p, _ in contacts.values())
    correction = np.zeros(len(joint_dofs))
    jac = np.zeros((3, model.nv))
    for i, (point, body) in enumerate(contacts.values()):
        f = world_force / len(contacts) if allocated is None else allocated[i].copy()
        if allocated is None and denom > 1e-8:
            r = point[:2] - center[:2]
            f[:2] += moment * np.array([-r[1], r[0]]) / denom
        mujoco.mj_jac(model, data, jac, None, point, body)
        correction -= jac[:, joint_dofs].T @ f
    return np.clip(correction, -3., 3.)
