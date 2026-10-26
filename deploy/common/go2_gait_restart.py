"""One bounded stand/reentry attempt when a requested left gait stays static.

Uses joint motion, not an estimated base velocity. No DDS, motor publisher,
contact force, or simulated state is used. Explicitly enabled by the caller.
"""
from collections import deque
import numpy as np


class GaitRestart:
    def __init__(self):
        self.samples = deque()
        self.phase = 'monitor'
        self.attempted = False
        self.blocked = False
        self.count = 0
        self.sequence = 0
        self.event = None
        self.started = None
        self.stable_since = None
        self.initial = None
        self.metrics = {}
        self.last_sensor_tick = None

    @property
    def active(self):
        return self.phase in ('stand', 'exit_hold')

    def notify(self, event, now):
        self.sequence += 1
        self.event = dict(event=event, time=float(now), sequence=self.sequence)

    def reset_request(self):
        """Explicit X or a different direction permits one new attempt."""
        self.samples.clear()
        self.attempted = self.blocked = False
        if not self.active:
            self.phase = 'monitor'

    def cancel(self):
        self.samples.clear()
        self.phase = 'stopped'

    def observe(self, now, target, q, dq, gyro, sensor_tick=None):
        target, q, dq, gyro = [np.asarray(v, dtype=float) for v in (target, q, dq, gyro)]
        if (target.shape != (3,) or q.shape != (12,) or dq.shape != (12,) or gyro.shape != (3,)
                or not all(np.isfinite(v).all() for v in (target, q, dq, gyro)) or not np.isfinite(now)):
            raise ValueError('起步检测状态无效')
        if sensor_tick is not None:
            advancing = self.last_sensor_tick is None or sensor_tick > self.last_sensor_tick
            self.last_sensor_tick = sensor_tick
            if not advancing:
                self.samples.clear()
                return None
        left = target[1] >= .2 and abs(target[0]) < 1e-6 and abs(target[2]) < 1e-6
        if self.active or not left:
            self.samples.clear()
            return None
        if self.blocked:
            return 'blocked'
        if self.samples and not 0 < now-self.samples[-1][0] <= .05:
            self.samples.clear()
        self.samples.append((now, q.copy(), dq.copy(), float(np.max(np.abs(gyro)))))
        while self.samples and now-self.samples[0][0] > 1.:
            self.samples.popleft()
        if len(self.samples) < 20 or now-self.samples[0][0] < .8:
            return None
        q_std = float(np.std([v[1] for v in self.samples], axis=0).max())
        dq_rms = float(np.sqrt(np.mean(np.square([v[2] for v in self.samples]))))
        gyro_max = max(v[3] for v in self.samples)
        self.metrics = dict(joint_std_max_rad=q_std, joint_speed_rms_rad_s=dq_rms,
                            gyro_max_rad_s=gyro_max, window_seconds=now-self.samples[0][0])
        if q_std >= .003 or dq_rms >= .1 or gyro_max >= .15:
            return None
        self.samples.clear()
        if self.attempted:
            self.blocked = True
            self.phase = 'blocked'
            self.notify('left_blocked', now)
            return 'blocked'
        return 'restart'

    def begin(self, now, command):
        if self.active or self.attempted:
            raise ValueError('本次左移已经尝试过起步恢复')
        self.initial = (np.asarray(command.target_pos_sdk).copy(),
                        np.asarray(command.kp_sdk).copy(), np.asarray(command.kd_sdk).copy())
        if (not all(v.shape == (12,) and np.isfinite(v).all() for v in self.initial)
                or np.any(self.initial[1] <= 0) or np.any(self.initial[2] < 0)):
            raise ValueError('起步恢复缺少有效的上一控制命令')
        self.started = now
        self.stable_since = None
        self.phase = 'stand'
        self.attempted = True
        self.count += 1
        self.notify('stand_begin', now)

    def stand_command(self, now, q, dq, stand, kp_stand, kd_stand, tilt, exiting=False):
        if not self.active:
            raise ValueError('起步恢复状态无效')
        elapsed = now-self.started
        if elapsed < 0 or not np.isfinite(elapsed):
            raise ValueError('起步恢复时间无效')
        r = float(np.clip(elapsed/2., 0., 1.));r = r*r*(3-2*r)
        target0, kp0, kd0 = self.initial
        kp = kp0*(1-r)+kp_stand*r
        kd = kd0*(1-r)+kd_stand*r
        target = (kp0*(1-r)*target0+kp_stand*r*stand)/kp
        stable = elapsed >= 2. and np.max(np.abs(q-stand)) <= .2 and np.max(np.abs(dq)) <= .2 and tilt <= .2
        self.stable_since = (now if self.stable_since is None else self.stable_since) if stable else None
        complete = self.stable_since is not None and now-self.stable_since >= .8
        if complete and self.phase != 'exit_hold':
            self.phase = 'exit_hold' if exiting else 'monitor'
            self.notify('stand_ready', now)
        if elapsed > 5. and not complete:
            raise ValueError('左移起步恢复未能在5秒内稳定站姿')
        return target, kp, kd, complete and not exiting

    def audit(self):
        return dict(phase=self.phase, attempted=self.attempted, blocked=self.blocked,
                    count=self.count, event=self.event, **self.metrics)
