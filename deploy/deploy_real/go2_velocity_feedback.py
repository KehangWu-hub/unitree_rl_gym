"""Optional SportModeState velocity observation and bounded velocity PI.

No DDS or motor publisher is initialized on import. The stance-force controller
is deliberately absent: this module changes only policy velocity commands.
"""
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from deploy.common.go2_velocity_control import VectorVelocityPI

PROFILE = Path(__file__).with_name('configs') / 'go2_velocity_pi.yaml'


def body_velocity(velocity, quaternion, frame):
    value = np.asarray(velocity, dtype=float)
    quat = np.asarray(quaternion, dtype=float)
    if (value.shape != (3,) or quat.shape != (4,) or not np.isfinite(value).all()
            or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-6):
        raise ValueError('速度或姿态反馈无效')
    if frame == 'body':
        return value.copy()
    if frame != 'world':
        raise ValueError('速度坐标系未经确认；请先完成只读观测')
    w, x, y, z = quat / np.linalg.norm(quat)
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return rotation.T @ value


class SportVelocitySource:
    """Freshness uses receipt time AND strictly advancing publisher stamps."""
    def __init__(self, timeout=.1):
        self.timeout = timeout
        self._lock = threading.Lock()
        self._row = None
        self._stamp = None
        self._error = '尚未收到机身速度'
        self.subscriber = None

    def start(self):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        self.subscriber = ChannelSubscriber('rt/sportmodestate', SportModeState_)
        self.subscriber.Init(self.callback, 10)

    def close(self):
        if self.subscriber is not None:
            self.subscriber.Close()

    def callback(self, message):
        received = time.monotonic()
        try:
            stamp = int(message.stamp.sec) * 1000000000 + int(message.stamp.nanosec)
            velocity = np.asarray(message.velocity, dtype=float)
            yaw = float(message.yaw_speed)
            if (stamp <= 0 or not 0 <= int(message.stamp.nanosec) < 1000000000
                    or int(message.error_code) != 0 or velocity.shape != (3,)
                    or not np.isfinite(velocity).all() or not np.isfinite(yaw)
                    or np.max(np.abs(velocity)) > 3 or abs(yaw) > 5):
                raise ValueError('机身速度报文无效或超出观测范围')
            with self._lock:
                if self._stamp is not None and stamp <= self._stamp:
                    # Duplicates must not extend the lifetime of an old estimate.
                    return
                self._stamp = stamp
                self._row = {'received': received, 'stamp_ns': stamp,
                             'velocity_raw': velocity.tolist(), 'yaw_speed_raw': yaw}
                self._error = ''
        except (ValueError, TypeError, AttributeError, OverflowError):
            with self._lock:
                self._error = '机身速度报文无效'

    def snapshot(self, now):
        with self._lock:
            row = None if self._row is None else dict(self._row)
            error = self._error
        if error or row is None:
            raise ValueError(error or '尚未收到机身速度')
        age = now - row['received']
        if not 0 <= age <= self.timeout:
            raise ValueError('机身速度超时或时间异常；PI反馈已失效')
        row['age_seconds'] = age
        return row


class KinematicVelocitySource:
    """LowState-only FK estimate; no transport, contacts or simulated base velocity."""
    def __init__(self, config, timeout=.1):
        from deploy.common.go2_velocity_estimator import KinematicVelocityEstimator
        from legged_gym import LEGGED_GYM_ROOT_DIR
        model = Path(LEGGED_GYM_ROOT_DIR) / 'resources/robots/go2/mjcf/scene_flat.xml'
        implementation = Path(__file__).parents[1] / 'common/go2_velocity_estimator.py'
        self.estimator = KinematicVelocityEstimator(model, config.data['policy']['sdk_joint_names'], height_scale=.001)
        self.fingerprint = hashlib.sha256(model.read_bytes()+model.with_name('go2.xml').read_bytes()+implementation.read_bytes()+Path(__file__).read_bytes()).hexdigest()
        self.timeout, self._row, self._tick = timeout, None, None
        self._error = '尚未收到用于速度估计的LowState'

    def start(self):
        pass

    def close(self):
        pass

    def observe(self, state, now):
        try:
            if state is None or not 0 <= now-state.timestamp <= self.timeout:
                raise ValueError('运动学速度LowState超时')
            tick = (getattr(state, 'feedback', None) or {}).get('state_tick')
            if not isinstance(tick, (int, np.integer)) or tick < 0:
                raise ValueError('运动学速度缺少有效LowState tick')
            if self._tick is not None and tick < self._tick:
                raise ValueError('运动学速度LowState tick倒退')
            if tick == self._tick:
                return  # A repeated sample cannot refresh its timestamp.
            dt = .02 if self._row is None else state.timestamp-self._row['received']
            if not 0 < dt <= self.timeout:
                raise ValueError('运动学速度采样周期异常')
            velocity = self.estimator.estimate(state.joint_pos_sdk, state.joint_vel_sdk,
                state.quaternion_wxyz, state.angular_velocity, dt)
            gyro = np.asarray(state.angular_velocity, dtype=float)
            if not np.isfinite(velocity).all() or np.max(abs(velocity)) > 3:
                raise ValueError('运动学速度估计无效或超出范围')
            self._row = dict(received=float(state.timestamp), stamp_ns=int(tick)*1000000,
                velocity_raw=velocity.tolist(), yaw_speed_raw=float(gyro[2]),
                estimator_sha256=self.fingerprint, foot_weights=self.estimator.weights.tolist())
            self._tick, self._error = tick, ''
        except (ValueError, TypeError, AttributeError) as error:
            self._error = str(error)
            raise ValueError(self._error) from error

    def snapshot(self, now):
        if self._error or self._row is None:
            raise ValueError(self._error)
        age = now-self._row['received']
        if not 0 <= age <= self.timeout:
            raise ValueError('运动学速度LowState超时')
        return dict(self._row, age_seconds=age)


class VelocityFeedback:
    """Shadow never modifies commands. PI is an explicit field-test candidate."""
    def __init__(self, mode, command_limits, frame=None, source="sport", config=None, speed_profile="legacy"):
        if mode not in ('shadow', 'pi'):
            raise ValueError('无效的速度反馈模式')
        if frame not in (None, 'body', 'world') or (mode == 'pi' and frame is None):
            raise ValueError('启用PI前必须确认速度坐标系并指定--velocity-frame')
        self.profile_sha256 = hashlib.sha256(PROFILE.read_bytes()).hexdigest()
        with PROFILE.open() as stream:
            self.profile = yaml.safe_load(stream)
        self.mode, self.frame = mode, frame
        self.limits = np.asarray(command_limits, dtype=float)
        if self.limits.shape != (3,) or not np.isfinite(self.limits).all() or np.any(self.limits <= 0):
            raise ValueError('无效的策略速度上限')
        if source not in ('sport', 'kinematic'):
            raise ValueError('未知速度反馈来源')
        if source == 'kinematic' and (config is None or frame != 'body'):
            raise ValueError('运动学速度需要机器人配置和body坐标系')
        if speed_profile not in ('legacy', 'uniform'):
            raise ValueError('未知速度配置')
        if speed_profile == 'uniform':
            self.profile.update(self.profile['uniform'])
            if not (np.allclose(self.limits, [.5,.4,.5], atol=1e-7, rtol=0) or np.allclose(self.limits, [.6,.5,.6], atol=1e-7, rtol=0)):
                raise ValueError('uniform速度PI需要匹配的运行时指令限幅')
            self.limits = np.asarray(self.profile['command_limits'],dtype=float)
            if not np.allclose(self.limits,[.6,.5,.6],atol=1e-7,rtol=0):
                raise ValueError('uniform补偿指令限幅必须为0.6/0.5/0.6')
        self.source_name, self.speed_profile = source, speed_profile
        self.source = (KinematicVelocitySource(config, self.profile['feedback_timeout'])
                       if source == 'kinematic' else SportVelocitySource(self.profile['feedback_timeout']))
        values = [self.profile[k] for k in ('kp', 'ki', 'filter_seconds', 'command_slew',
                                             'feedback_timeout', 'max_yaw_disagreement')]
        target_limits = np.asarray(self.profile['target_limits'], dtype=float)
        correction = np.asarray(self.profile['max_correction'], dtype=float)
        if (not np.isfinite(values).all() or min(values) <= 0
                or target_limits.shape != (3,) or correction.shape != (3,)
                or not np.isfinite(target_limits).all() or not np.isfinite(correction).all()
                or np.any(target_limits <= 0) or np.any(target_limits > self.limits)
                or np.any(correction <= 0) or self.profile['feedback_timeout'] > .1):
            raise ValueError('PI候选参数无效或超出原控制边界')
        self.pi = VectorVelocityPI(np.c_[-self.limits, self.limits],
                                   kp=self.profile['kp'], ki=self.profile['ki'],
                                   filter_seconds=self.profile['filter_seconds'],
                                   slew=self.profile['command_slew'])
        self.audit = {'mode': mode, 'frame': frame or 'unverified', 'valid': False}
        self.last_time = None
        self.last_target = np.zeros(3)
        self.lowering = ControlledLowering()

    def reset(self):
        self.pi.reference.fill(0)
        self.pi.command.fill(0)
        self.pi.integral.fill(0)
        self.pi.filtered = None

    def apply(self, target, state, now):
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all() or not np.isfinite(now):
            raise ValueError('速度目标或时间无效')
        self.audit = {'mode': self.mode, 'frame': self.frame or 'unverified',
                      'source': self.source_name, 'speed_profile': self.speed_profile,
                      'profile_sha256': self.profile_sha256,
                      'target': target.tolist(), 'valid': False}
        try:
            if self.source_name == 'kinematic':
                self.source.observe(state, now)
            row = self.source.snapshot(now)
            self.audit.update(row)
            # Without confirmed frame shadow retains raw data only.
            if self.frame is None:
                self.audit.update(valid=True, applied=target.tolist())
                return target.copy()
            velocity = body_velocity(row['velocity_raw'], state.quaternion_wxyz, self.frame)
            gyro = np.asarray(state.angular_velocity, dtype=float)
            if gyro.shape != (3,) or not np.isfinite(gyro).all():
                raise ValueError('IMU角速度无效')
            measured = np.r_[velocity[:2], gyro[2]]
            if abs(row['yaw_speed_raw'] - gyro[2]) > self.profile['max_yaw_disagreement']:
                raise ValueError('里程计与IMU转向速度不一致')
            dt = .02 if self.last_time is None else now - self.last_time
            if not 0 < dt <= .05:
                raise ValueError('速度PI更新周期异常')
            clipped = np.clip(target, -np.asarray(self.profile['target_limits']),
                              np.asarray(self.profile['target_limits']))
            if not np.any(target) or not np.array_equal(np.sign(target), np.sign(self.last_target)):
                self.reset()
            self.last_target = target.copy()
            self.last_time = now
            if np.any(target):
                correction = np.asarray(self.profile['max_correction'])
                self.pi.bounds = np.c_[np.maximum(-self.limits, clipped - correction),
                                       np.minimum(self.limits, clipped + correction)]
                candidate = self.pi.update(clipped, measured, dt,
                    active_axes=(clipped != 0) if self.speed_profile == 'uniform' else None)
            else:
                # X, keyboard expiry and C clear every integral immediately.
                candidate = np.zeros(3)
            applied = candidate if self.mode == 'pi' else target
            self.audit.update(valid=True, measured_body=measured.tolist(),
                              candidate_target=clipped.tolist(), candidate=candidate.tolist(),
                              applied=applied.tolist(), integral=self.pi.integral.tolist())
            return applied.copy()
        except ValueError as error:
            self.reset()
            self.last_time = None
            self.last_target.fill(0)
            self.audit.update(error=str(error), applied=(target if self.mode == 'shadow' else np.zeros(3)).tolist())
            if self.mode == 'pi':
                raise
            return target.copy()

    def normal_exit_command(self, trial, state, now):
        if self.lowering.exit_initial is None:
            self.reset()
        return self.lowering.command(trial, state, now)


class ControlledLowering:
    """Joint-only stand transition and lowering shared by explicit candidates."""
    def __init__(self, stand_blend_seconds=2., live_policy_blend=False):
        if not np.isfinite(stand_blend_seconds) or stand_blend_seconds < 2.:
            raise ValueError("站姿过渡时间至少2秒")
        self.stand_blend_seconds = float(stand_blend_seconds)
        self.exit_initial = None
        self.exit_trim = np.zeros(12)
        self.live_policy_blend = bool(live_policy_blend)

    def command(self, trial, state, now):
        """Blend to static stand before lowering; no contact/velocity truth."""
        from deploy.deploy_real.deploy_real_go2 import Go2Command, Go2Mode, quaternion_to_roll_pitch
        if state is None or not 0 <= now-state.timestamp <= .1:
            raise ValueError('受控退出LowState通信超时')
        if trial.last_time is not None and not 0 <= now-trial.last_time <= .05:
            raise ValueError('受控退出发布循环超时')
        q, dq = np.asarray(state.joint_pos_sdk), np.asarray(state.joint_vel_sdk)
        tau_est = np.asarray((getattr(state, 'feedback', None) or {}).get('joint_tau_est_sdk'))
        if (q.shape != (12,) or dq.shape != (12,) or tau_est.shape != (12,)
                or not np.isfinite(q).all() or not np.isfinite(dq).all() or not np.isfinite(tau_est).all()):
            raise ValueError('受控退出关节反馈无效')
        roll, pitch = quaternion_to_roll_pitch(state.quaternion_wxyz)
        from deploy.deploy_real.deploy_real_go2 import require_safe_attitude
        require_safe_attitude(roll, pitch, trial.config.safety.get('fall_tilt_limit_rad', .35))
        enforce = trial.config.software_motion_limits_enabled
        if enforce and (max(abs(roll),abs(pitch)) > .35 or np.any(abs(dq)>trial.policy_velocity_limits)
                or np.any(q<trial.lower-.02) or np.any(q>trial.upper+.02)
                or np.max(abs(tau_est))>trial.torque_limit):
            raise ValueError('受控退出姿态、关节或反馈力矩超限')
        elapsed = now-trial.started
        begin = trial.normal_exit['elapsed']+2.
        if self.exit_initial is None:
            old = trial.last_policy_output
            initial_tau = old.kp_sdk*(old.target_pos_sdk-q)-old.kd_sdk*dq
            target = q+(initial_tau+old.kd_sdk*dq)/old.kp_sdk
            if enforce:
                target = np.clip(target,trial.lower,trial.upper)
            self.exit_initial = Go2Command(Go2Mode.POLICY,target,old.kp_sdk,old.kd_sdk)
            self.exit_last_time = now
            trial.controller.set_velocity_command(np.zeros(3))
        ratio = float(np.clip((elapsed-begin)/self.stand_blend_seconds,0,1));ratio=ratio*ratio*(3-2*ratio)
        if self.live_policy_blend and ratio < 1.:
            # Keep zero-command balance feedback while transferring support.
            # A frozen gait target can unload a leg during a multi-second blend.
            if trial.policy_time is None or now-trial.policy_time >= trial.config.control_dt-1e-7:
                trial.controller.set_velocity_command(np.zeros(3))
                live = trial.controller.step(now,state)
                if live.mode is not Go2Mode.POLICY or live.reason:
                    raise ValueError(live.reason or '退出过渡策略已停止')
                trial.policy_command = trial.limit_policy_pd(live,state)
                trial.policy_time = now
                trial.inference_count += 1
            self.exit_initial = trial.policy_command
        lowering = float(np.clip((elapsed-trial.lower_begin)/8.,0,1))
        if enforce and ratio >= 1. and np.max(abs(dq)) > 2.:
            raise ValueError('静态站姿或降低关节速度超过2rad/s')
        static_kp,static_kd = 60.,5.
        initial = self.exit_initial
        kp = initial.kp_sdk*(1-ratio)+static_kp*ratio
        kd = initial.kd_sdk*(1-ratio)+static_kd*ratio
        desired = trial.blend(trial.stand,trial.lowered,lowering)
        if not enforce:
            target = (initial.kp_sdk*(1-ratio)*initial.target_pos_sdk+static_kp*ratio*desired)/kp
            trial.last_time = now
            if elapsed >= trial.duration:
                trial.stop(now,'测试时长结束')
                return Go2Command(Go2Mode.DAMPING,np.zeros(12),np.zeros(12),np.full(12,3.),trial.reason)
            return Go2Command(Go2Mode.FIX_STAND,target,kp,kd)
        dt = now-self.exit_last_time
        if elapsed >= trial.settle_begin:
            # Slow bounded correction only after lowering has finished. It is
            # deliberately absent during gait, stand blending and lowering.
            trim_target = np.clip(desired-q,-.04,.04)
            self.exit_trim += np.clip(trim_target-self.exit_trim,-.01*dt,.01*dt)
        self.exit_last_time = now
        static_target = np.clip(desired+self.exit_trim,trial.lower,trial.upper)
        target = (initial.kp_sdk*(1-ratio)*initial.target_pos_sdk+static_kp*ratio*static_target)/kp
        target = np.clip(target,trial.lower,trial.upper)
        if ratio >= 1.:
            target = np.clip(target,q-.24,q+.24)
            target = np.clip(target,trial.lower,trial.upper)
        tau = np.clip(kp*(target-q)-kd*dq,-trial.policy_pd_limit,trial.policy_pd_limit)
        target = np.clip(q+(tau+kd*dq)/kp,trial.lower,trial.upper)
        if np.max(abs(kp*(target-q)-kd*dq))>trial.policy_pd_limit+1e-5:
            raise ValueError('受控退出PD力矩限幅失败')
        if ratio >= 1. and np.max(abs(target-q)) > trial.tracking_tolerance:
            raise ValueError('静态站姿或降低跟踪误差超过0.25弧度')
        trial.last_time = now
        if elapsed>=trial.duration:
            legs=q.reshape(4,3)
            if (max(abs(roll),abs(pitch))>.2 or np.max(abs(dq))>.2
                    or np.max(abs(q-trial.lowered))>.15 or np.any(abs(legs[:,0])>.5)
                    or np.any(legs[:,1]<1.1) or np.any(legs[:,1]>1.6) or np.any(legs[:,2]>-2.4)):
                raise ValueError('受控降低末态姿态、静止或跟踪检查失败')
            trial.stop(now,'测试时长结束')
            return Go2Command(Go2Mode.DAMPING,np.zeros(12),np.zeros(12),np.full(12,3.),trial.reason)
        return Go2Command(Go2Mode.FIX_STAND,target,kp,kd)


def validate_feedback_readiness(path, policy_sha256, config_sha256, source="sport", estimator_sha256=None, speed_profile=None, profile_sha256=None):
    """Verify post-handoff publication in a shadow log; not speed calibration."""
    path = Path(path)
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    active = [r for r in rows if r.get('mode') == 'policy'
              and r.get('feedback', {}).get('test_elapsed', -1) >= 8.]
    if not active:
        raise ValueError('速度前置记录没有接管后的策略观测')
    times, receipts, stamps = [], [], []
    for row in active:
        f = row.get('feedback', {})
        v = f.get('velocity_feedback', {})
        if (f.get('policy_sha256') != policy_sha256 or f.get('policy_config_sha256') != config_sha256
                or v.get('mode') != 'shadow' or v.get('valid') is not True
                or not 0 <= v.get('age_seconds', float('inf')) <= .1):
            raise ValueError('速度前置记录存在反馈缺失、过期或版本不匹配')
        if (v.get('source', 'sport') != source
                or (estimator_sha256 is not None and v.get('estimator_sha256') != estimator_sha256)
                or (speed_profile is not None and v.get('speed_profile') != speed_profile)
                or (profile_sha256 is not None and v.get('profile_sha256') != profile_sha256)):
            raise ValueError('速度前置记录的来源、估计器或参数版本不匹配')
        velocity = np.asarray(v.get('velocity_raw'), dtype=float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError('速度前置记录包含无效速度')
        times.append(float(f['test_elapsed']))
        receipts.append(float(v['received']))
        stamps.append(int(v['stamp_ns']))
    times, receipts, stamps = map(np.asarray, (times, receipts, stamps))
    if (not np.isfinite(times).all() or not np.isfinite(receipts).all()
            or times[-1] - times[0] < 5 or np.any(np.diff(times) < 0)
            or np.max(np.diff(times)) > .1 or np.any(np.diff(receipts) < 0)
            or np.any(np.diff(stamps) < 0) or len(np.unique(stamps)) < 50
            or np.max(np.diff(np.unique(receipts))) > .1):
        raise ValueError('速度前置记录未达到接管后连续5秒、时间戳推进和最大100ms间隔要求')
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
            'post_handoff_seconds': float(times[-1] - times[0]),
            'scope': 'publication continuity only; coordinate frame and speed accuracy require field verification'}
