"""Go2真机阻尼、逐关节、站立验收及限时策略试跑。"""

import argparse
import copy
from contextlib import contextmanager
from collections import deque
import gc
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from legged_gym import LEGGED_GYM_ROOT_DIR
from deploy.deploy_real.deploy_real_go2 import (
    Go2Command, Go2Controller, Go2DDSTransport, Go2Mode, Go2RealConfig, Go2Recorder,
    KeyboardVelocity, TerminalKeys, _array, quaternion_to_roll_pitch, model_joint_velocity_limits,
)
from deploy.deploy_real.go2_watchdog import COMMAND_TIMEOUT, default_socket, request


class VerifiedTransport(Go2DDSTransport):
    def __init__(self, config, publish=True):
        self._observed_lowcmd = None
        self._observed_lowcmd_count = 0
        self._commission_config = config
        self._sent_signatures = deque(maxlen=128)
        self.guard_command_source = False
        self.foreign_command = False
        super().__init__(config, publish=publish)
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
        self._command_observer = ChannelSubscriber(config.data['dds']['lowcmd_topic'], LowCmd_)
        self._command_observer.Init(self._on_command, 10)

    @staticmethod
    def signature(q, kp, kd):
        return np.asarray([q, kp, kd], dtype=np.float32).tobytes()

    def prepare_factory_handoff(self):
        """已确认normal时只初始化发布器，不写消息；释放服务后立即保持。"""
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
        if self._publisher is not None:
            raise ValueError('发布器已初始化')
        self._publisher = ChannelPublisher(self._commission_config.data['dds']['lowcmd_topic'], LowCmd_)
        self._publisher.Init()
        self._publish = True

    def send(self, command):
        signature = self.signature(command.target_pos_sdk, command.kp_sdk, command.kd_sdk)
        with self._lock:
            self._sent_signatures.append(signature)
        super().send(command)

    def _on_command(self, message):
        observed = {
            'time': time.monotonic(), 'crc_valid': self._crc.Crc(message) == message.crc,
            'target_pos_sdk': [float(m.q) for m in message.motor_cmd[:12]],
            'kp_sdk': [float(m.kp) for m in message.motor_cmd[:12]],
            'kd_sdk': [float(m.kd) for m in message.motor_cmd[:12]],
            'mode_sdk': [int(m.mode) for m in message.motor_cmd[:12]],
        }
        with self._lock:
            if self.guard_command_source and (not observed['crc_valid'] or
                    self.signature(observed['target_pos_sdk'], observed['kp_sdk'], observed['kd_sdk']) not in self._sent_signatures):
                self.foreign_command = True
            self._observed_lowcmd_count += 1
            observed['count'] = self._observed_lowcmd_count
            self._observed_lowcmd = observed

    def _on_state(self, message):
        # 不把CRC错误帧更新为“新鲜状态”；持续错误会触发状态超时。
        if self._crc.Crc(message) == message.crc:
            from deploy.deploy_real.deploy_real_go2 import Go2State
            state = Go2State(time.monotonic(), np.asarray(message.imu_state.quaternion, dtype=np.float32),
                             np.asarray(message.imu_state.gyroscope, dtype=np.float32),
                             np.asarray([m.q for m in message.motor_state[:12]], dtype=np.float32),
                             np.asarray([m.dq for m in message.motor_state[:12]], dtype=np.float32))
            feedback = {
                'state_tick': int(message.tick),
                'joint_tau_est_sdk': [float(m.tau_est) for m in message.motor_state[:12]],
                'motor_mode_sdk': [int(m.mode) for m in message.motor_state[:12]],
                'motor_lost_sdk': [int(m.lost) for m in message.motor_state[:12]],
                'motor_temperature_sdk': [int(m.temperature) for m in message.motor_state[:12]],
                # 原样保留SDK字段；未经有效性/单位核对，不能作为实测承重或停机判据。
                'foot_force_raw': [int(value) for value in message.foot_force],
                'foot_force_est_raw': [int(value) for value in message.foot_force_est],
                'imu_accelerometer_raw': [float(value) for value in message.imu_state.accelerometer],
            }
            with self._lock:
                state.feedback = feedback
                self._state = state

    def snapshot(self):
        # 一次锁内复制状态和其反馈，保持相同帧；旧订阅状态也兼容。
        from deploy.deploy_real.deploy_real_go2 import Go2State
        with self._lock:
            if self._state is None:
                return None
            state = Go2State(self._state.timestamp, self._state.quaternion_wxyz.copy(),
                             self._state.angular_velocity.copy(), self._state.joint_pos_sdk.copy(),
                             self._state.joint_vel_sdk.copy())
            feedback = getattr(self._state, 'feedback', None)
            state.feedback = dict(feedback) if feedback is not None else None
            if state.feedback is not None:
                # DDS回显只证明本机收到该消息，不是机器人执行确认。
                state.feedback['observed_lowcmd'] = getattr(self, '_observed_lowcmd', None)
            return state


def check_prone(config, state, now, initial=None):
    """这是已观测趴卧姿态的验收范围，不是新的策略/机械关节限位。"""
    if state is None or not 0 <= now - state.timestamp <= min(0.1, config.safety['state_timeout']):
        raise ValueError('LowState通信超时或时间无效')
    q = _array(state.joint_pos_sdk, 'q')
    dq = _array(state.joint_vel_sdk, 'dq')
    roll, pitch = quaternion_to_roll_pitch(state.quaternion_wxyz)
    if not np.isfinite([roll, pitch]).all() or max(abs(roll), abs(pitch)) > 0.2:
        raise ValueError('趴卧姿态倾斜超过0.2弧度')
    policy_q = q[config.policy_to_sdk].reshape(4, 3)
    if (np.any(np.abs(policy_q[:, 0]) > 0.5)
            or np.any(policy_q[:, 1] < 1.1) or np.any(policy_q[:, 1] > 1.6)
            or np.any(policy_q[:, 2] < -2.9) or np.any(policy_q[:, 2] > -2.6)):
        raise ValueError('不符合已观测的趴卧姿态；禁止站立状态低增益接管')
    if np.max(np.abs(dq)) > (0.2 if initial is None else 2.0):
        raise ValueError('关节速度超限')
    if initial is not None and np.max(np.abs(q - initial)) > 0.04:
        raise ValueError('实际关节位移超过0.04弧度')
    return q, dq


def check_resting_for_restore(config, state, now):
    """仅用于原厂恢复；现场已确认接触地面，兼容急停后腿的自然展开。"""
    if state is None or not 0 <= now - state.timestamp <= .1:
        raise ValueError('恢复前LowState通信超时')
    q, dq = _array(state.joint_pos_sdk, 'q'), _array(state.joint_vel_sdk, 'dq')
    roll, pitch = quaternion_to_roll_pitch(state.quaternion_wxyz)
    limbs = q[config.policy_to_sdk].reshape(4, 3)
    failures = []
    if max(abs(roll),abs(pitch)) > .2:
        failures.append(f'倾角roll={roll:.3f},pitch={pitch:.3f}rad，要求绝对值<=0.2')
    if np.max(np.abs(dq)) > .2:
        failures.append(f'关节速度峰值={np.max(np.abs(dq)):.3f}rad/s，要求<=0.2')
    for index, name in enumerate(config.data['policy']['joint_names']):
        leg, joint = divmod(index,3)
        value = limbs[leg,joint]
        lower, upper = ((-.7,.7),(.8,1.6),(-2.9,-2.4))[joint]
        if not lower <= value <= upper:
            failures.append(f'{name}={value:.3f}rad，低姿态要求[{lower:g},{upper:g}]')
    if failures:
        raise ValueError('恢复前未确认静止低姿态：'+'；'.join(failures))
    return q, dq


def wait_resting_for_restore(config, reader, stable_seconds=1., timeout=15., clock=None, sleep=None):
    """仅订阅状态，等待连续静止低姿态，不切模式、不发送控制命令。"""
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    deadline = clock() + timeout
    stable_since = None
    last_error = '尚未收到有效状态'
    while True:
        state = reader.snapshot()
        now = clock()  # 必须在复制状态之后读取时钟。
        try:
            check_resting_for_restore(config,state,now)
            stable_since = now if stable_since is None else stable_since
            if now - stable_since >= stable_seconds:
                return state
        except ValueError as error:
            stable_since = None
            last_error = str(error)
        if now >= deadline:
            raise ValueError(f'等待连续{stable_seconds:g}秒静止低姿态超时；原厂模式未恢复。最后未通过项：{last_error}')
        sleep(config.control_dt)


def check_standing(config, state, now):
    if state is None or not 0 <= now - state.timestamp <= .1:
        raise ValueError('站立接管LowState超时')
    q, dq = _array(state.joint_pos_sdk, 'q'), _array(state.joint_vel_sdk, 'dq')
    roll, pitch = quaternion_to_roll_pitch(state.quaternion_wxyz)
    limbs = q[config.policy_to_sdk].reshape(4, 3)
    if max(abs(roll), abs(pitch)) > .2 or np.max(np.abs(dq)) > .2:
        raise ValueError('原厂站立姿态或静止检查未通过')
    if (np.any(np.abs(limbs[:, 0]) > .4) or np.any(limbs[:, 1] < .4) or np.any(limbs[:, 1] > 1.2)
            or np.any(limbs[:, 2] < -2.) or np.any(limbs[:, 2] > -1.)):
        raise ValueError('不符合原厂站立接管范围')
    if np.any(q[config.policy_to_sdk] < np.asarray(config.safety['joint_lower']) - .02) or np.any(q[config.policy_to_sdk] > np.asarray(config.safety['joint_upper']) + .02):
        raise ValueError('原厂站姿不在模型状态范围内')
    return q, dq


class ProneTest:
    def __init__(self, config, joint, amplitude, duration, motor, response=False, joint_check=False):
        if not joint_check and joint not in ('FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint'):
            raise ValueError('真机受限测试只允许髋关节')
        if joint_check and (not motor or joint not in config.data['policy']['joint_names']):
            raise ValueError('逐关节验收必须选择配置中的Go2驱动关节')
        if not math.isfinite(amplitude) or not 0 < abs(amplitude) <= (0.02 if joint_check else 0.01):
            raise ValueError('真机幅度超出当前测试范围或为零')
        if not math.isfinite(duration) or not 1 <= duration <= 3:
            raise ValueError('真机测试时长必须在1至3秒内')
        if response and (not motor or amplitude != 0.005 or duration != 3):
            raise ValueError('响应测试固定3秒，目标幅度0.005弧度')
        if joint_check and (amplitude != 0.02 or duration != 3 or response):
            raise ValueError('逐关节验收固定0.02弧度、3秒，单独使用')
        self.config, self.amplitude, self.duration, self.motor = config, amplitude, duration, motor
        self.response = response
        self.joint_check = joint_check
        self.policy_joint = config.data['policy']['joint_names'].index(joint)
        self.joint = int(config.policy_to_sdk[self.policy_joint])
        canonical = [f'{leg}_{kind}_joint' for leg in ('FR', 'FL', 'RR', 'RL') for kind in ('hip', 'thigh', 'calf')]
        if self.joint != canonical.index(joint):
            raise ValueError('关节SDK映射与宇树Go2顺序不一致')
        self.initial = None
        self.started = None
        self.stopped = None
        self.reason = ''
        self.recover_range = False

    def stop(self, now, reason):
        if not self.reason:
            self.reason, self.stopped = str(reason), now

    def arm(self, now, state):
        if self.reason or self.started is not None:
            raise ValueError('测试已经启动或停止，不能重新启动')
        self.initial, _ = check_prone(self.config, state, now)
        if self.response or self.joint_check:
            _array((getattr(state, 'feedback', None) or {}).get('joint_tau_est_sdk'), 'tau_est')
        lower = self.config.safety['joint_lower'][self.policy_joint]
        if self.joint_check and self.initial[self.joint] < lower:
            if not self.config.data['policy']['joint_names'][self.policy_joint].endswith('_calf_joint') or lower - self.initial[self.joint] > 0.15:
                raise ValueError('起点超出允许的趴卧小腿过渡范围')
            self.recover_range = True
        self.initial = self.initial.copy()
        self.started = now

    def phase(self, now):
        if not (self.response or self.joint_check) or self.started is None:
            return 'legacy', 0.0
        elapsed = now - self.started
        # 基线0.4s、正向0.8s、基线0.4s、负向0.8s、基线0.6s。
        for name, begin, end, sign in (('positive', .4, 1.2, 1), ('negative', 1.6, 2.4, -1)):
            if begin <= elapsed < end:
                if self.recover_range and sign == -1:
                    return 'positive_repeat', self.amplitude * np.sin(np.pi * (elapsed - begin) / (end - begin)) ** 2
                return name, sign * self.amplitude * np.sin(np.pi * (elapsed - begin) / (end - begin)) ** 2
        return 'baseline', 0.0

    def output(self, now, state):
        if self.started is None and not self.reason:
            return None
        if not self.reason:
            try:
                q, dq = check_prone(self.config, state, now, self.initial)
                elapsed = now - self.started
                if elapsed >= self.duration:
                    self.stop(now, '测试时长结束')
                target = self.initial.copy()
                kp = np.zeros(12, dtype=np.float32)
                kd = np.full(12, 0.5, dtype=np.float32)
                if self.motor:
                    if self.response or self.joint_check:
                        _, offset = self.phase(now)
                        target[self.joint] += offset
                    else:
                        target[self.joint] += self.amplitude * np.sin(np.pi * min(elapsed / self.duration, 1)) ** 2
                    kp[self.joint] = 40.0 if self.joint_check else 20.0 if self.response else 5.0
                    lower = self.config.safety['joint_lower'][self.policy_joint]
                    upper = self.config.safety['joint_upper'][self.policy_joint]
                    target_lower = float(self.initial[self.joint]) if self.recover_range else lower
                    if not target_lower <= target[self.joint] <= upper:
                        raise ValueError('主动关节目标超过允许范围')
                if self.response or self.joint_check:
                    tau = _array((getattr(state, 'feedback', None) or {}).get('joint_tau_est_sdk'), 'tau_est')
                    if abs(tau[self.joint]) > (2.0 if self.joint_check else 0.5) or np.max(np.abs(q - self.initial)) > (0.03 if self.joint_check else 0.01) or np.max(np.abs(dq)) > 0.5:
                        raise ValueError('响应测试反馈力矩/实际位移/速度超限')
                    if self.recover_range and q[self.joint] < self.initial[self.joint] - 0.003:
                        raise ValueError('小腿向更折叠方向偏移超过0.003弧度')
                if np.max(np.abs(kp * (target - q) - kd * dq)) > (0.2 if self.response else 1.0):
                    raise ValueError('估算PD力矩超过响应测试上限')
                if not self.reason:
                    return Go2Command(Go2Mode.MOTOR_TEST if self.motor else Go2Mode.PASSIVE, target, kp, kd)
            except (ValueError, TypeError, AttributeError) as error:
                self.stop(now, str(error))
        return Go2Command(Go2Mode.DAMPING, np.zeros(12), np.zeros(12), np.full(12, 3.0), self.reason)


class StandCheck:
    """无策略的趴卧归位、站立、FixStand增益保持及受控降低；不进入Policy。"""

    duration = 17.0
    motor = True
    joint = -1
    recover_range = False
    torque_limit = 12.0
    hold_tolerance = .2
    tracking_tolerance = .25
    settle_tracking_tolerance = .15

    def __init__(self, config, start_standing=False):
        self.config = config
        self.start_standing = start_standing
        self.duration = 14. if start_standing else 17.
        self.entry_end = 0. if start_standing else 3.
        self.hold_begin = 5. if start_standing else 8.
        self.lower_begin = self.hold_begin + 2.
        self.settle_begin = self.lower_begin + 5.
        self.initial = self.started = self.stopped = None
        self.reason = ''
        self.last_time = None
        self.hold_samples = 0
        self.last_hold_tick = None
        self.hold_first = self.hold_last = None
        self.handoff_started = False
        self.handoff_delay_seconds = 0.
        # SDK顺序，来自宇树go2_stand_example.py的卧倒目标。
        self.crouch = np.array([0, 1.36, -2.65, 0, 1.36, -2.65,
                               -.2, 1.36, -2.65, .2, 1.36, -2.65], dtype=np.float32)
        self.lowered = self.crouch.copy()
        if start_standing:
            # 实体降低至-2.65目标会因负载偏差触碰模型下限；目标向内收0.1rad。
            self.lowered[2::3] = -2.55
        self.stand = config.default_joint_pos[config.sdk_to_policy].copy()
        self.hold_kp = _array(config.data['fsm']['stand_kp'], 'stand_kp')[config.sdk_to_policy]
        self.hold_kd = _array(config.data['fsm']['stand_kd'], 'stand_kd')[config.sdk_to_policy]
        if np.any(self.hold_kp <= 0) or np.any(self.hold_kp > 60) or np.any(self.hold_kd < .5) or np.any(self.hold_kd > 5):
            raise ValueError('FixStand增益不在已验证的站立验收范围内')
        self.factory_stand = np.tile([0., .67, -1.3], 4).astype(np.float32)
        self.lower = np.asarray(config.safety['joint_lower'])[config.sdk_to_policy]
        self.upper = np.asarray(config.safety['joint_upper'])[config.sdk_to_policy]
        for target in (self.crouch, self.lowered, self.factory_stand, self.stand):
            if np.any(target < self.lower) or np.any(target > self.upper):
                raise ValueError('站立验收目标超出原有模型范围')

    def stop(self, now, reason):
        if not self.reason:
            self.reason, self.stopped = str(reason), now

    def arm(self, now, state):
        if self.reason or self.started is not None:
            raise ValueError('测试已经启动或停止，不能重新启动')
        q, _ = (check_standing if self.start_standing else check_prone)(self.config, state, now)
        _array((getattr(state, 'feedback', None) or {}).get('joint_tau_est_sdk'), 'tau_est')
        outside = q < self.lower
        calves = np.array([i % 3 == 2 for i in range(12)])
        if np.any(outside & ~calves) or np.any(q < self.lower - .15) or np.any(q > self.upper):
            raise ValueError('站立起点超出允许的趴卧小腿过渡范围')
        self.initial = q.copy()
        self.started = now

    def begin_handoff(self, now):
        """释放前只验证起点；释放确认后才开始完整的运动计时。"""
        if not self.start_standing or self.started is None or self.reason or self.last_time is not None or self.handoff_started:
            raise ValueError('不能重置已运行或停止的接管时钟')
        if not math.isfinite(now) or now < self.started:
            raise ValueError('接管起始时间无效')
        self.handoff_delay_seconds = now - self.started
        self.started = now
        self.handoff_started = True

    @staticmethod
    def blend(start, end, ratio):
        r = float(np.clip(ratio, 0, 1))
        return start + (end - start) * (r * r * (3 - 2 * r))

    def phase(self, now):
        e = 0 if self.started is None else now - self.started
        return ('range_entry' if e < self.entry_end else 'rise' if e < self.hold_begin else
                'hold' if e < self.lower_begin else 'lower' if e < self.settle_begin else 'settle'), 0.0

    def profile(self, now):
        e = now - self.started
        phase = self.phase(now)[0]
        if phase == 'range_entry':
            target = self.blend(self.initial, self.crouch, e / 3)
        elif phase == 'rise':
            # 先走官方示例站姿，再过渡到训练站姿，避免只靠端点直线插值。
            target = (self.blend(self.initial, self.stand, e / 5) if self.start_standing else
                      self.blend(self.crouch, self.factory_stand, (e - 3) / 3) if e < 6 else
                      self.blend(self.factory_stand, self.stand, (e - 6) / 2))
        elif phase == 'hold':
            target = self.stand.copy()
        elif phase == 'lower':
            target = self.blend(self.stand, self.lowered, (e - self.lower_begin) / 5)
        else:
            target = self.lowered.copy()
        # 起身/降低使用官方示例60/5；保持使用配置FixStand增益。
        kp, kd = np.full(12, 60.), np.full(12, 5.)
        if phase == 'hold':
            r = np.clip(e - self.hold_begin, 0, 1)
            kp = kp * (1 - r) + self.hold_kp * r
            kd = kd * (1 - r) + self.hold_kd * r
        elif phase == 'lower':
            r = np.clip((e - self.lower_begin) / .5, 0, 1)
            kp = self.hold_kp * (1 - r) + kp * r
            kd = self.hold_kd * (1 - r) + kd * r
        return target, kp, kd

    def output(self, now, state):
        if self.started is None and not self.reason:
            return None
        if not self.reason:
            try:
                if state is None or not 0 <= now - state.timestamp <= .1:
                    raise ValueError('LowState通信超时')
                if self.last_time is not None and not 0 <= now - self.last_time <= .05:
                    raise ValueError(f'站立验收循环超时: {(now - self.last_time) * 1000:.1f}ms')
                q, dq = _array(state.joint_pos_sdk, 'q'), _array(state.joint_vel_sdk, 'dq')
                tau = _array((getattr(state, 'feedback', None) or {}).get('joint_tau_est_sdk'), 'tau_est')
                roll, pitch = quaternion_to_roll_pitch(state.quaternion_wxyz)
                if max(abs(roll), abs(pitch)) > .35 or np.max(np.abs(dq)) > 2:
                    raise ValueError('站立验收姿态/速度超限')
                e = now - self.started
                phase = self.phase(now)[0]
                state_lower = np.minimum(self.initial - .003, self.lower - .02) if e < self.entry_end else self.lower - .02
                if np.any(q < state_lower) or np.any(q > self.upper + .02):
                    raise ValueError('站立验收实际关节超限；过渡未及时进入模型范围')
                target, kp, kd = self.profile(now)
                target_lower = np.minimum(self.initial, self.lower) if e < self.entry_end else self.lower
                if np.any(target < target_lower - 1e-7) or np.any(target > self.upper):
                    raise ValueError('站立验收目标超限')
                if np.max(np.abs(kp * (target - q) - kd * dq)) > self.torque_limit or np.max(np.abs(tau)) > self.torque_limit:
                    raise ValueError(f'站立验收PD/反馈估算力矩超过{self.torque_limit:g}Nm')
                if np.max(np.abs(target - q)) > self.tracking_tolerance:
                    raise ValueError('站立验收跟踪误差超过0.25弧度')
                if self.hold_begin + 1 <= e < self.lower_begin and np.max(np.abs(q - self.stand)) <= self.hold_tolerance and max(abs(roll), abs(pitch)) <= .2:
                    tick = (getattr(state, 'feedback', None) or {}).get('state_tick')
                    if tick is None:
                        raise ValueError('站立验收缺少状态tick')
                    if tick != self.last_hold_tick:
                        self.hold_samples += 1
                        self.last_hold_tick = tick
                        self.hold_first = state.timestamp if self.hold_first is None else self.hold_first
                        self.hold_last = state.timestamp
                elif self.hold_begin + 1 <= e < self.lower_begin:
                    self.hold_samples = 0
                    self.hold_first = self.hold_last = None
                if phase == 'lower' and (self.hold_samples < 50 or self.hold_first is None or self.hold_last - self.hold_first < .8):
                    raise ValueError('FixStand增益站立保持未达到验收要求')
                if e >= self.duration:
                    if self.start_standing:
                        # 此时是低姿态，随后零Kp阻尼才放到地面；不声称已接触地面。
                        limbs = q[self.config.policy_to_sdk].reshape(4, 3)
                        if (max(abs(roll), abs(pitch)) > .2 or np.max(np.abs(dq)) > .2
                                or np.any(np.abs(limbs[:, 0]) > .5)
                                or np.any(limbs[:, 1] < 1.1) or np.any(limbs[:, 1] > 1.6)
                                or np.any(limbs[:, 2] > -2.4)):
                            raise ValueError('降低后的低姿态/静止检查未通过')
                    else:
                        check_prone(self.config, state, now)
                    if np.max(np.abs(q - self.lowered)) > self.settle_tracking_tolerance:
                        raise ValueError('受控降低未接近卧倒目标')
                    self.stop(now, '测试时长结束')
                self.last_time = now
                if not self.reason:
                    return Go2Command(Go2Mode.FIX_STAND, target, kp, kd)
            except (ValueError, TypeError, AttributeError) as error:
                self.stop(now, str(error))
        return Go2Command(Go2Mode.DAMPING, np.zeros(12), np.zeros(12), np.full(12, 3.), self.reason)


def model_joint_velocity_ceiling(config):
    """兼容旧诊断脚本的摘要；实际检查使用逐关节上限。"""
    return float(np.min(model_joint_velocity_limits(config)))


def resolve_action_step_mode(mode, moving):
    if mode is None:
        return 'limit' if moving else 'reject'
    if mode not in ('reject', 'limit') or (mode == 'limit' and not moving):
        raise ValueError('动作步长限幅仅用于运动策略流程')
    return mode


def keyboard_timing(forward_speed, keyboard_axis, motion_seconds=None, continuous=False):
    """输入窗口末尾、累计运动预算、控制总时长；退出后另发一秒阻尼。"""
    if continuous:
        if keyboard_axis != 'all' or forward_speed != .4 or motion_seconds is not None:
            raise ValueError('持续控制仅用于all，不能同时指定运动预算')
        return float('inf'), float('inf'), float('inf')
    if motion_seconds is not None:
        if keyboard_axis != 'all' or forward_speed != .4 or motion_seconds not in (4, 10):
            raise ValueError('运动时长选择仅允许all统一键盘的4秒或10秒档')
    budget = float(motion_seconds) if motion_seconds is not None else 4. if forward_speed == .4 else 2.
    window_end = 20. + budget - 4. if forward_speed == .4 else 14.
    return window_end, budget, window_end + 9.


class PolicyCheck(StandCheck):
    """原厂站姿接管；持续交互或短时策略检查共用同一保护链路。"""

    def __init__(self, config, readiness_log, policy=None, forward=False, forward_speed=.1, keyboard=False, keyboard_axis="forward", yaw_speed=.3, turn_forward_speed=0., keyboard_motion_seconds=None, walk_speed=.3, action_step_mode=None, continuous=False, left_speed=.3, speed_profile="legacy"):
        if config.data.get('format_version') == 2:
            if speed_profile != 'legacy' or walk_speed != .3 or left_speed != .3 or yaw_speed != .3 or turn_forward_speed not in (0., .3):
                raise ValueError('当前HIM仅验证0.30平移和0.30行进转向配置')
            if action_step_mode is None:
                action_step_mode = 'limit'
        if speed_profile not in ('legacy', 'uniform'):
            raise ValueError('未知速度配置')
        uniform = speed_profile == 'uniform'
        if uniform and (not keyboard or keyboard_axis != 'all' or not continuous
                        or walk_speed != .5 or left_speed != .4 or yaw_speed != .5 or turn_forward_speed != .5):
            raise ValueError('uniform需要all持续控制、前后0.5、侧移0.4、行进转向0.5/0.5')
        self.speed_profile = speed_profile
        if continuous and not keyboard:
            raise ValueError("持续控制需要键盘运动模式")
        if forward_speed not in (.1, .3, .4):
            raise ValueError('前进速度只允许0.10、0.30或键盘0.40m/s限时档')
        if (keyboard and (not forward or forward_speed not in (.3, .4))) or (forward_speed == .4 and not keyboard):
            raise ValueError('键盘验收仅允许0.30或0.40m/s前进档')
        if keyboard_axis not in ("forward", "backward", "lateral", "yaw", "all") or (keyboard_axis != "forward" and (not keyboard or forward_speed != .4)):
            raise ValueError("后退/侧移/转向/统一键盘仅允许0.4档受限流程")
        if yaw_speed not in (.3,.4,.5) or (yaw_speed != .3 and keyboard_axis != "yaw" and not uniform):
            raise ValueError("0.50rad/s仅允许转向键盘候选")
        if turn_forward_speed not in ((0.,.3,.5) if uniform else (0.,.3)) or (turn_forward_speed and keyboard_axis not in ("yaw", "all")):
            raise ValueError("边走边转仅允许yaw或all轴加0.30m/s前进")
        if not uniform and keyboard_axis == "all" and (yaw_speed != .3 or turn_forward_speed != .3):
            raise ValueError("统一键盘固定0.30m/s平移与0.30rad/s行进转向")
        if keyboard_motion_seconds is not None and not keyboard:
            raise ValueError('运动时长选择需要统一键盘流程')
        if walk_speed not in (.3, .5) or (walk_speed != .3 and (not keyboard or keyboard_axis != "all")):
            raise ValueError("W前进速度仅允许统一键盘0.30或0.50m/s")
        if walk_speed > float(config.data["command_limits"][0]):
            raise ValueError("W前进速度超过配置指令上限")
        action_step_mode = resolve_action_step_mode(action_step_mode, forward or config.data.get("format_version") == 2)
        if walk_speed > .3 and action_step_mode != "limit":
            raise ValueError("提高前进速度必须启用动作步长限幅，禁止使用旧的直接拒绝模式")
        if left_speed not in (.3,.4) or (left_speed != .3 and (not keyboard or keyboard_axis != 'all' or not continuous or action_step_mode != 'limit')):
            raise ValueError('A速度候选仅支持all模式0.40且启用动作限幅')
        self.left_speed = left_speed
        self.left_candidate_used = False
        self.controlled_lowering = None
        self.walk_speed = walk_speed
        self.turn_forward_speed = turn_forward_speed
        self.yaw_speed = yaw_speed
        self.keyboard_axis = keyboard_axis
        self.keyboard_axis_index = {"forward":0,"backward":0,"lateral":1,"yaw":2,"all":None}[keyboard_axis]
        self.keyboard_keys = {"forward":("w",),"backward":("s",),"lateral":("a","d"),"yaw":("q","e"),"all":("q","w","e","a","s","d")}[keyboard_axis]
        self.keyboard = keyboard
        self.continuous = continuous
        self.keyboard_window_end, self.keyboard_budget, _ = keyboard_timing(forward_speed, keyboard_axis, keyboard_motion_seconds, continuous)
        self.keyboard_timeout = .6 if forward_speed == .4 else .25
        self.keyboard_input = KeyboardVelocity([.3 if keyboard_axis in ('backward', 'all') else forward_speed,.30,yaw_speed], timeout=self.keyboard_timeout) if keyboard else None
        self.keyboard_used = 0.
        self.keyboard_last_time = None
        self.keyboard_budget_time = None
        self.keyboard_velocity = 0.
        self.keyboard_command = np.zeros(3, dtype=np.float32)
        self.keyboard_event = None
        self.keyboard_sequence = 0
        self.forward_speed = forward_speed
        self.policy_velocity_limits = (model_joint_velocity_limits(config)[config.sdk_to_policy]
                                       if forward else np.full(12, 2.))
        self.policy_velocity_limit = float(np.min(self.policy_velocity_limits))  # 历史摘要字段

        self.entry_blend_seconds = .5
        with open(config.policy_path, 'rb') as stream:
            self.policy_sha256 = hashlib.sha256(stream.read()).hexdigest()
        commissioning = config.data.get('commissioning', {})
        self.policy_pd_limit = float(commissioning.get('policy_pd_limit_nm', 10.5))
        self.torque_limit = float(commissioning.get('feedback_torque_limit_nm', 12.0))
        model_ceiling = float(np.min(config.torque_limits * config.safety['torque_limit_ratio']))
        if (not np.isfinite([self.policy_pd_limit, self.torque_limit]).all()
                or not 0 < self.policy_pd_limit < self.torque_limit <= model_ceiling):
            raise ValueError('验收力矩参数必须满足0<PD限幅<反馈保护<=模型力矩保护')
        self._validate_readiness(readiness_log, config, forward)
        super().__init__(config, start_standing=True)
        if keyboard_axis in ('backward', 'lateral'):
            # 新动作末态沿用整个降低阶段的跟踪界限。静止、姿态、实际关节
            # 和PD/反馈力矩仍独立检查；不改变目标、增益或已有动作的末态判据。
            self.settle_tracking_tolerance = self.tracking_tolerance
        self.policy_begin = 7.
        self.forward = bool(forward)
        self.lower_begin = self.keyboard_window_end + 2. if keyboard else 12. if forward else 9.
        self.settle_begin = self.lower_begin + 5.
        self.duration = self.settle_begin + 2.
        controller_config = config
        if left_speed > .3:
            controller_config = copy.deepcopy(config)
            controller_config.data['command_limits'][1] = left_speed
        self.controller = Go2Controller(controller_config, policy=policy, action_step_mode=action_step_mode)
        self.controller.policy_pd_limit = self.policy_pd_limit
        if action_step_mode == "limit":
            self.controller.action_step_limit = min(self.controller.safety.max_action_delta,
                8. * config.control_dt / float(config.data["actions"]["scale"]))
        self.controller.validate_policy()  # 加载与首轮推理在DDS初始化/释放原厂模式前完成。
        self.controller.set_velocity_command(np.zeros(3))
        self.policy_command = None
        self.last_policy_output = None
        self.policy_time = None
        self.exit_start = None
        self.inference_count = 0
        self.normal_exit = None
        self.velocity_feedback = None
        self.gait_restart = None
        self.policy_reentry_at = None
        self.him_scheduled_exit = self.lower_begin if config.data.get("format_version") == 2 and not continuous else None
        if self.him_scheduled_exit is not None:
            self.duration += 14.

    def _validate_readiness(self, readiness_log, config, forward):
        """Physical entry requires physical evidence; simulation subclasses never publish DDS."""
        with open(readiness_log, encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
            if config.data.get("format_version") == 2:
                from deploy.common.him_qualification import require_sim2sim
                require_sim2sim(config)
                active = [r for r in rows if r.get("mode") == ("policy" if forward else "fix_stand")]
                if any(r.get("simulation_only") for r in rows) or not active:
                    raise ValueError("HIM现场前置记录不能使用仿真结果")
                for row in active:
                    feedback = row.get("feedback", {})
                    key = "policy_config_sha256" if forward else "control_config_sha256"
                    if feedback.get(key) != config.source_sha256 or (forward and feedback.get("policy_sha256") != self.policy_sha256):
                        raise ValueError("HIM前置记录的模型/配置摘要不匹配")
            ready = assess_policy(rows, config) if forward else assess_stand(rows, config)
        expected = 'zero_velocity_policy_trial_passed' if forward else 'stand_hold_lower_passed'
        if ready['result'] != expected:
            raise ValueError(f'前置{"零速度策略" if forward else "站立"}验收未通过: {ready}')
        if forward:
            prior = [row for row in rows if row.get('mode') == 'policy']
            if not prior or not all(row.get('feedback', {}).get('policy_entry_blend_seconds') == self.entry_blend_seconds
                    and row.get('feedback', {}).get('policy_pd_limit_nm') == self.policy_pd_limit
                    and row.get('feedback', {}).get('policy_feedback_limit_nm', 12.) == self.torque_limit for row in prior):
                raise ValueError('前置零速度日志未验证当前接管曲线和PD限幅，拒绝前进')

    def stop(self, now, reason):
        observer = getattr(self, 'velocity_feedback', None)
        if observer is not None:
            observer.reset()
        if getattr(self, 'gait_restart', None) is not None:
            self.gait_restart.cancel()
        super().stop(now, reason)

    def arm(self, now, state):
        super().arm(now, state)
        self.controller.request_fix_stand(now, state)

    def begin_handoff(self, now):
        if self.controller.fsm.mode is not Go2Mode.FIX_STAND:
            raise ValueError('策略接管尚未准备FixStand')
        super().begin_handoff(now)
        self.controller.fsm.mode_started_at = now

    def phase(self, now):
        e = 0 if self.started is None else now - self.started
        if self.policy_begin <= e < self.lower_begin:
            return 'policy', 0.
        return super().phase(now)

    def blend_policy_entry(self, now, command):
        elapsed = now-self.started-self.policy_begin
        if self.policy_reentry_at is not None:
            elapsed = now-self.policy_reentry_at
        ratio = np.clip(elapsed / self.entry_blend_seconds, 0., 1.)
        ratio = ratio * ratio * (3 - 2 * ratio)
        kp = self.hold_kp * (1 - ratio) + command.kp_sdk * ratio
        kd = self.hold_kd * (1 - ratio) + command.kd_sdk * ratio
        # 按Kp加权目标，令实际PD为保持PD与策略PD的凸组合。
        target = (self.hold_kp * (1 - ratio) * self.stand + command.kp_sdk * ratio * command.target_pos_sdk) / kp
        return Go2Command(Go2Mode.POLICY, target, kp, kd)

    def limit_policy_pd(self, command, state):
        q, dq = _array(state.joint_pos_sdk, 'q'), _array(state.joint_vel_sdk, 'dq')
        order = self.config.policy_to_sdk
        target = self.controller.safety.limit_pd_target(
            command.target_pos_sdk[order], q[order], dq[order], command.kp_sdk[order],
            command.kd_sdk[order], self.policy_pd_limit)
        return Go2Command(Go2Mode.POLICY, self.controller._to_sdk(target), command.kp_sdk, command.kd_sdk)

    def checked_policy_output(self, command, state, now):
        command = self.limit_policy_pd(command, state)
        q, dq = _array(state.joint_pos_sdk, 'q'), _array(state.joint_vel_sdk, 'dq')
        tau = _array((getattr(state, 'feedback', None) or {}).get('joint_tau_est_sdk'), 'tau_est')
        pd = command.kp_sdk*(command.target_pos_sdk-q)-command.kd_sdk*dq
        if np.max(np.abs(pd)) > self.torque_limit or np.max(np.abs(tau)) > self.torque_limit:
            raise ValueError(f'策略试跑力矩保护: PD={np.max(np.abs(pd)):.3f}Nm,反馈={np.max(np.abs(tau)):.3f}Nm,上限{self.torque_limit:g}Nm')
        self.last_policy_output = command
        self.last_time = now
        return command

    def restart_output(self, now, state, tilt):
        recovery = self.gait_restart
        if np.max(np.abs(state.joint_vel_sdk)) > 2.:
            raise ValueError('左移起步恢复关节速度超过2rad/s')
        self.controller.set_velocity_command(np.zeros(3))
        self.keyboard_command.fill(0.)
        self.keyboard_velocity = 0.
        self.keyboard_last_time = now
        target, kp, kd, ready = recovery.stand_command(
            now, state.joint_pos_sdk, state.joint_vel_sdk, self.stand, self.hold_kp,
            self.hold_kd, tilt, exiting=self.normal_exit is not None)
        if ready:
            # Same zero previous action as a fresh policy entry, only after
            # the physical stand transition and stability checks have passed.
            self.controller.reset_policy_state()
            # Inference was paused; the publisher clock self.last_time remains checked.
            self.controller.last_step_time = None
            self.policy_time = None
            self.policy_reentry_at = now
            if self.velocity_feedback is not None:
                self.velocity_feedback.reset()
            return None
        if self.velocity_feedback is not None:
            self.velocity_feedback.apply(np.zeros(3), state, now)
        command = self.checked_policy_output(Go2Command(Go2Mode.POLICY, target, kp, kd), state, now)
        if now-recovery.started >= 2. and np.max(np.abs(command.target_pos_sdk-state.joint_pos_sdk)) > self.tracking_tolerance:
            raise ValueError('左移恢复站姿跟踪误差超过0.25弧度')
        return command

    def request_normal_exit(self, now):
        """归零两秒后沿用5秒降低、2秒稳定，再一秒阻尼退出。"""
        if not self.continuous or self.normal_exit is not None:
            return False
        if self.reason or self.started is None or self.phase(now)[0] != 'policy' or self.policy_command is None:
            return False
        elapsed = now - self.started
        self.keyboard_input.feed('x', now)
        self.keyboard_command.fill(0.)
        self.keyboard_velocity = 0.
        self.keyboard_window_end = elapsed
        self.lower_begin = elapsed + 2.
        self.settle_begin = self.lower_begin + 5.
        self.duration = self.settle_begin + 2.
        if self.velocity_feedback is not None and self.velocity_feedback.mode == 'pi':
            self.lower_begin = elapsed + 8.
            self.settle_begin = self.lower_begin + 8.
            self.duration = self.settle_begin + 3.
        if self.left_candidate_used:
            from deploy.deploy_real.go2_velocity_feedback import ControlledLowering
            self.controlled_lowering = ControlledLowering()
            self.lower_begin = elapsed + 8.
            self.settle_begin = self.lower_begin + 8.
            self.duration = self.settle_begin + 3.
        if self.gait_restart is not None and self.gait_restart.active:
            self.lower_begin = elapsed + 3.
            self.settle_begin = self.lower_begin + 5.
            self.duration = self.settle_begin + 2.
        if self.config.data.get("format_version") == 2:
            self._configure_him_exit(elapsed)
        self.normal_exit = {'key':'c', 'elapsed':elapsed, 'lower_begin':self.lower_begin,
                            'duration':self.duration}
        return True

    def _configure_him_exit(self, elapsed):
        from deploy.deploy_real.go2_velocity_feedback import ControlledLowering
        self.controlled_lowering = ControlledLowering(stand_blend_seconds=4.)
        self.lower_begin = elapsed + 10.  # brake 2, stand blend 4, hold 4
        self.settle_begin = self.lower_begin + 8.
        self.duration = self.settle_begin + 5.
        self.normal_exit = {'key': 'scheduled', 'elapsed': elapsed,
                            'lower_begin': self.lower_begin, 'duration': self.duration}

    def feed_keys(self, keys, now):
        # 同一次终端读取中，归零键优先于方向键和自动重复输入。
        stop_key = 'c' if 'c' in keys else next((key for key in keys if key in ('x', ' ')), None)
        if stop_key is not None:
            self.feed_key(stop_key, now)
        else:
            for key in keys:
                self.feed_key(key, now)

    def feed_key(self, key, now):
        if not self.keyboard or key not in (*self.keyboard_keys,'x',' ','c'):
            return
        elapsed = None if self.started is None else now - self.started
        self.account_keyboard_budget(now)
        if key == 'c':
            accepted = self.request_normal_exit(now)
            self.keyboard_sequence += 1
            self.keyboard_event = {'key':key,'elapsed':elapsed,'sequence':self.keyboard_sequence,
                                   'accepted':accepted,'reason':'' if accepted else 'normal_exit_unavailable'}
            return
        if self.normal_exit is not None:
            self.keyboard_sequence += 1
            self.keyboard_event = {'key':key,'elapsed':elapsed,'sequence':self.keyboard_sequence,
                                   'accepted':False,'reason':'normal_exit_in_progress'}
            return
        if key in self.keyboard_keys and (elapsed is None or not 8 <= elapsed < self.keyboard_window_end or self.keyboard_used >= self.keyboard_budget):
            self.keyboard_sequence += 1
            self.keyboard_event = {'key':key,'elapsed':elapsed,'sequence':self.keyboard_sequence,'accepted':False,
                                   'reason':'budget_exhausted' if self.keyboard_used >= self.keyboard_budget else 'outside_window'}
            return
        self.account_keyboard_budget(now)
        self.keyboard_input.feed(key,now)
        if self.gait_restart is not None and key != 'a':
            self.gait_restart.reset_request()
        if key in ('x',' '):
            self.keyboard_velocity = 0.
            self.keyboard_command.fill(0.)
        self.keyboard_sequence += 1
        self.keyboard_event = {'key':key,'elapsed':elapsed,'sequence':self.keyboard_sequence,'accepted':True}

    def account_keyboard_budget(self,now):
        if self.keyboard_budget_time is not None and abs(self.keyboard_velocity) > 0:
            self.keyboard_used += max(0.,now-self.keyboard_budget_time)
        self.keyboard_budget_time = now

    def keyboard_profile(self, now):
        dt = 0. if self.keyboard_last_time is None else max(0., now-self.keyboard_last_time)
        self.account_keyboard_budget(now)
        self.keyboard_last_time = now
        elapsed = now-self.started
        if self.keyboard_axis == "all":
            return self.unified_keyboard_profile(now, elapsed, dt)
        target = self.keyboard_input.sample(now)[self.keyboard_axis_index] if 8 <= elapsed < self.keyboard_window_end and self.keyboard_used < self.keyboard_budget else 0.
        if target == 0:
            self.keyboard_velocity = 0.
        elif target*self.keyboard_velocity < 0:
            self.keyboard_velocity = 0.  # 反向先归零，下一推理周期才向另一方向加速。
        else:
            self.keyboard_velocity = float(np.sign(target)*min(abs(target),abs(self.keyboard_velocity)+dt))
        velocity=np.zeros(3,dtype=np.float32)
        velocity[self.keyboard_axis_index]=self.keyboard_velocity
        if self.turn_forward_speed and self.keyboard_velocity:
            velocity[0]=self.turn_forward_speed*abs(self.keyboard_velocity)/self.yaw_speed
        return velocity

    def unified_keyboard_profile(self, now, elapsed, dt):
        target = self.keyboard_input.sample(now) if 8 <= elapsed < self.keyboard_window_end and self.keyboard_used < self.keyboard_budget else np.zeros(3)
        if target[2]:
            target[0] = self.turn_forward_speed * abs(target[2]) / self.yaw_speed
        elif target[0] > 0:
            target[0] = self.walk_speed
        elif target[1] > 0:
            target[1] = self.left_speed
        if self.speed_profile == 'uniform':
            if target[0] < 0: target[0] = -self.walk_speed
            if target[1] < 0: target[1] = -self.left_speed
        # 每次只执行最后一个方向键；换方向先归零一个策略周期，再重新加速。
        if not np.any(target) or (np.any(self.keyboard_command) and
                not np.array_equal(np.sign(target), np.sign(self.keyboard_command))):
            self.keyboard_command.fill(0.)
        else:
            magnitude = self.walk_speed if target[0] > 0 and not target[2] else self.left_speed if target[1] > 0 else .3
            if self.speed_profile == 'uniform':
                magnitude = float(np.max(np.abs(target)))
            ratio = min(1., (self.keyboard_velocity + dt) / magnitude)
            self.keyboard_command = np.asarray(target * ratio, dtype=np.float32)
        self.keyboard_velocity = float(np.max(np.abs(self.keyboard_command)))
        if (self.left_speed > .3 and self.keyboard_command[1] > .3+1e-6) or (self.speed_profile == 'uniform' and np.any(self.keyboard_command)):
            self.left_candidate_used = True
        if self.keyboard_command[0] < 0 or self.keyboard_command[1] != 0:
            # 仅实际执行后退/侧移后沿用这些动作既有的降低末态门槛。
            self.settle_tracking_tolerance = self.tracking_tolerance
        return self.keyboard_command.copy()

    def velocity_profile(self, now):
        """前1秒零速度、2秒前进（0.3秒平滑加减速）、后2秒零速度。"""
        if self.keyboard:
            return self.keyboard_profile(now)
        velocity = np.zeros(3, dtype=np.float32)
        if self.forward and self.started is not None:
            elapsed = now - self.started - self.policy_begin - 1.
            if 0 <= elapsed < 2:
                ratio = min(elapsed / .3, (2 - elapsed) / .3, 1.)
                velocity[0] = self.forward_speed * ratio * ratio * (3 - 2 * ratio)
        return velocity

    def profile(self, now):
        if self.phase(now)[0] == 'lower':
            if self.exit_start is None:
                raise ValueError('缺少策略退出目标，拒绝降低')
            return (self.blend(self.exit_start, self.lowered, (now - self.started - self.lower_begin) / 5),
                    np.full(12, 60.), np.full(12, 5.))
        return super().profile(now)

    def output(self, now, state):
        if (self.him_scheduled_exit is not None and self.started is not None
                and self.normal_exit is None and now-self.started >= self.him_scheduled_exit):
            self._configure_him_exit(self.him_scheduled_exit-2.)
        phase = self.phase(now)[0]
        if (self.started is not None and not self.reason and self.normal_exit is not None
                and self.controlled_lowering is not None
                and now-self.started >= self.normal_exit['elapsed']+2.):
            try:
                return self.controlled_lowering.command(self, state, now)
            except (ValueError, TypeError, AttributeError, RuntimeError) as error:
                self.stop(now, str(error))
        if (self.started is not None and not self.reason and self.normal_exit is not None
                and self.velocity_feedback is not None and self.velocity_feedback.mode == 'pi'
                and now-self.started >= self.normal_exit['elapsed']+2.):
            try:
                return self.velocity_feedback.normal_exit_command(self, state, now)
            except (ValueError, TypeError, AttributeError, RuntimeError) as error:
                self.stop(now, str(error))
        if self.started is not None and not self.reason and phase == 'policy':
            try:
                if state is None or not 0 <= now - state.timestamp <= .1:
                    raise ValueError('LowState通信超时')
                if self.last_time is not None and not 0 <= now - self.last_time <= .05:
                    raise ValueError('策略发布循环超时')
                q, dq = _array(state.joint_pos_sdk, 'q'), _array(state.joint_vel_sdk, 'dq')
                tau = _array((getattr(state, 'feedback', None) or {}).get('joint_tau_est_sdk'), 'tau_est')
                roll, pitch = quaternion_to_roll_pitch(state.quaternion_wxyz)
                if max(abs(roll), abs(pitch)) > .35:
                    raise ValueError('策略试跑姿态超限')
                exceeded = np.flatnonzero(np.abs(dq) > self.policy_velocity_limits)
                if exceeded.size:
                    joint = int(exceeded[0])
                    name = self.config.data['policy']['sdk_joint_names'][joint]
                    raise ValueError(f'策略关节速度{name}={dq[joint]:.3f}rad/s超过{self.policy_velocity_limits[joint]:g}rad/s')
                if np.any(q < self.lower - .02) or np.any(q > self.upper + .02):
                    raise ValueError('策略试跑实际关节超限')
                if self.gait_restart is not None and self.gait_restart.active:
                    command = self.restart_output(now, state, max(abs(roll), abs(pitch)))
                    if command is not None:
                        return command
                if self.policy_command is None:
                    if self.hold_samples < 50 or self.hold_first is None or self.hold_last - self.hold_first < .8:
                        raise ValueError('FixStand稳定保持不足，拒绝运行策略')
                    self.controller.fsm.stand_target(now)
                    if not self.controller.request_policy(now):
                        raise ValueError('策略状态切换失败')
                if self.policy_time is None or now - self.policy_time >= self.config.control_dt - 1e-7:
                    target_velocity = self.velocity_profile(now)
                    if self.gait_restart is not None and self.normal_exit is None:
                        event = self.gait_restart.observe(now, target_velocity, q, dq, state.angular_velocity,
                                                          (state.feedback or {}).get('state_tick'))
                        if event == 'restart':
                            self.gait_restart.begin(now, self.last_policy_output)
                            command = self.restart_output(now, state, max(abs(roll), abs(pitch)))
                            if command is not None:
                                return command
                        elif event == 'blocked':
                            target_velocity = np.zeros(3)
                    if self.velocity_feedback is not None:
                        target_velocity = self.velocity_feedback.apply(target_velocity, state, now)
                    self.controller.set_velocity_command(target_velocity)
                    command = self.controller.step(now, state)
                    if command.mode is not Go2Mode.POLICY or command.reason:
                        raise ValueError(command.reason or '策略控制器已停止')
                    self.policy_command, self.policy_time = command, now
                    self.inference_count += 1
                return self.checked_policy_output(self.blend_policy_entry(now, self.policy_command), state, now)
            except (ValueError, TypeError, AttributeError, RuntimeError) as error:
                self.stop(now, str(error))
        elif self.started is not None and not self.reason and phase == 'lower' and self.exit_start is None:
            try:
                q, dq = _array(state.joint_pos_sdk, 'q'), _array(state.joint_vel_sdk, 'dq')
                if self.policy_command is None:
                    raise ValueError('未执行策略，拒绝进入策略退出阶段')
                c = self.last_policy_output
                pd = c.kp_sdk * (c.target_pos_sdk - q) - c.kd_sdk * dq
                # 切换60/5时保持同一帧估算PD，随后5秒平滑降低。
                self.exit_start = q + (pd + 5 * dq) / 60
                if np.any(self.exit_start < self.lower) or np.any(self.exit_start > self.upper):
                    raise ValueError('策略退出目标超出模型范围')
            except (ValueError, TypeError, AttributeError) as error:
                self.stop(now, str(error))
        return super().output(now, state)


def mode_client():
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    client = MotionSwitcherClient()
    client.SetTimeout(3.0)
    client.Init()
    return client


def read_mode(client):
    code, mode = client.CheckMode()
    if code != 0 or not isinstance(mode, dict) or not isinstance(mode.get('name'), str):
        raise RuntimeError('无法确认原厂运动模式，拒绝继续')
    return mode['name']


def release_mode(client):
    previous = read_mode(client)
    if previous not in ('normal', ''):
        raise RuntimeError(f'未知原厂模式{previous!r}，拒绝自动切换')
    if previous:
        code, _ = client.ReleaseMode()
        if code != 0:
            raise RuntimeError(f'退出原厂运动模式失败: {code}')
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if read_mode(client) == '':
            return previous
        time.sleep(0.1)
    raise RuntimeError('原厂运动模式仍未退出；不会发布LowCmd')


def assess_response(rows, joint=3):
    """区分力矩反馈与编码器响应；始终不声称已验证实体腿的方向。"""
    result = {'result': 'inconclusive', 'physical_joint_direction_confirmed': False}
    if any(row.get('reason') not in ('', '测试时长结束') for row in rows):
        return dict(result, result='aborted', reason='测试存在提前停止或安全故障')
    active = [row for row in rows if row.get('mode') == 'motor_test']
    try:
        if not active or active[-1]['time'] - active[0]['time'] < 2.8:
            raise ValueError('没有完整3秒响应记录')
        q = np.asarray([row['joint_pos_sdk'][joint] for row in active])
        target = np.asarray([row['target_pos_sdk'][joint] for row in active])
        tau = np.asarray([row['feedback']['joint_tau_est_sdk'][joint] for row in active])
        if not np.isfinite([q, target, tau]).all():
            raise ValueError('角度或力矩含非有限值')
        if len({row['feedback']['state_tick'] for row in active}) < 100:
            raise ValueError('有效状态帧不足')
        phase = np.asarray([row['feedback']['test_phase'] for row in active])
        relative_time = np.asarray([row['time'] - active[0]['time'] for row in active])
        # 激励后的返回/接触滞后不作为传感器静止噪声；只取首段基线。
        baseline = (phase == 'baseline') & (relative_time < .4)
        offset = target - target[0]
        amplitude = float(np.max(np.abs(offset)))
        if amplitude < .004 or amplitude > .02001:
            raise ValueError('响应目标幅度不符合诊断范围')
        transition = any(row['feedback'].get('range_transition', False) for row in active)
        positive = (phase == 'positive') & (offset >= .8 * amplitude)
        negative = ((phase == 'positive_repeat') & (offset >= .8 * amplitude) if transition else
                    (phase == 'negative') & (offset <= -.8 * amplitude))
        if min(np.sum(mask) for mask in (baseline, positive, negative)) < 20:
            raise ValueError('基线或正负目标峰值窗口不足')
        echoes = [row['feedback'].get('observed_lowcmd') for row in active]
        echoes = [echo for echo in echoes if echo is not None]
        if not echoes or not all(echo['crc_valid'] for echo in echoes):
            raise ValueError('缺少有效DDS回显')
        tau_base = float(np.median(tau[baseline]))
        tau_noise = float(1.4826 * np.median(np.abs(tau[baseline] - tau_base)))
        q_base = float(np.median(q[baseline]))
        q_noise = float(np.ptp(q[baseline]))
        tau_threshold = max(0.05, 3 * tau_noise)
        q_threshold = max(0.001, 4 * q_noise)
        tau_changes = [float(np.median(tau[mask]) - tau_base) for mask in (positive, negative)]
        q_changes = [float(np.median(q[mask]) - q_base) for mask in (positive, negative)]
        force = tau_changes[0] >= tau_threshold and (tau_changes[1] >= tau_threshold if transition else tau_changes[1] <= -tau_threshold)
        encoder = q_changes[0] >= q_threshold and (q_changes[1] >= q_threshold if transition else q_changes[1] <= -q_threshold)
        if transition:
            returned = (phase == 'baseline') & (relative_time >= 2.7)
            if np.sum(returned) < 20 or abs(float(np.median(q[returned])) - q_base) > 2 * q_threshold:
                encoder = False
        label = ('force_and_encoder_response' if force and encoder else
                 'force_response_only' if force else 'encoder_response_only' if encoder else 'inconclusive')
        return dict(result, result=label, force_response=force, encoder_response=encoder,
                    tau_changes_nm=tau_changes, q_changes_rad=q_changes,
                    tau_threshold_nm=tau_threshold, q_threshold_rad=q_threshold,
                    baseline_tau_noise_nm=tau_noise, baseline_q_span_rad=q_noise,
                    direction_tested='towards_model_range_and_return' if transition else 'positive_and_negative',
                    range_transition=transition,
                    recorded_mean_hz=(len(active) - 1) / (active[-1]['time'] - active[0]['time']))
    except (KeyError, TypeError, ValueError, IndexError) as error:
        return dict(result, reason=str(error))


def assess_stand(rows, config):
    result = {'result': 'inconclusive', 'policy_executed': False}
    if any(r.get('mode') == 'policy' for r in rows):
        return dict(result, result='aborted', policy_executed=True, reason='日志包含策略控制')
    if any(row.get('reason') not in ('', '测试时长结束') for row in rows):
        return dict(result, result='aborted', reason='测试有提前停止或安全故障')
    active = [r for r in rows if r.get('mode') == 'fix_stand']
    try:
        standing = any(r.get('feedback', {}).get('stand_profile') == 'standing14' for r in active)
        duration, held_after = (14., 6.) if standing else (17., 9.)
        if not active or active[-1]['time'] - active[0]['time'] < duration - .2:
            raise ValueError(f'缺少完整{duration:g}秒站立验收记录')
        nominal = config.default_joint_pos[config.sdk_to_policy]
        held = [r for r in active if r['feedback']['test_phase'] == 'hold'
                and r['feedback']['test_elapsed'] >= held_after]
        good = [r for r in held if max(abs(np.asarray(r['joint_pos_sdk']) - nominal)) <= StandCheck.hold_tolerance
                and max(abs(x) for x in quaternion_to_roll_pitch(r['quaternion_wxyz'])) <= .2]
        if len({r['feedback']['state_tick'] for r in good}) < 50:
            raise ValueError('FixStand增益站立保持记录不足')
        if len(good) != len(held) or good[-1]['time'] - good[0]['time'] < .8:
            raise ValueError('稳定保持时长不足0.8秒')
        expected_kp = _array(config.data['fsm']['stand_kp'], 'stand_kp')[config.sdk_to_policy]
        expected_kd = _array(config.data['fsm']['stand_kd'], 'stand_kd')[config.sdk_to_policy]
        if not all(np.allclose(r['kp_sdk'], expected_kp) and np.allclose(r['kd_sdk'], expected_kd) for r in held):
            raise ValueError('保持增益与配置不一致')
        if not all(r['feedback'].get('observed_lowcmd', {}).get('crc_valid', False) for r in held):
            raise ValueError('站立保持DDS回显不足或CRC错误')
        final = rows[-1]
        state = type('RecordedState', (), {'timestamp': final['time'], 'joint_pos_sdk': final['joint_pos_sdk'],
                                         'joint_vel_sdk': final['joint_vel_sdk'], 'quaternion_wxyz': final['quaternion_wxyz']})()
        check_prone(config, state, final['time'])
        return dict(result, result='stand_hold_lower_passed', hold_unique_ticks=len({r['feedback']['state_tick'] for r in good}),
                    max_hold_error_rad=max(float(np.max(np.abs(np.asarray(r['joint_pos_sdk']) - nominal))) for r in held))
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        return dict(result, reason=str(error))


def assess_policy(rows, config):
    result = {'result': 'inconclusive', 'policy_executed': any(r.get('mode') == 'policy' for r in rows)}
    try:
        forward = any(r.get('feedback', {}).get('stand_profile') == 'forward19' for r in rows)
        if any(r.get('reason') not in ('', '测试时长结束') for r in rows):
            raise ValueError('提前停止或安全故障，不能认定试跑通过')
        policy = [r for r in rows if r.get('mode') == 'policy']
        forward_speed = policy[0].get('feedback', {}).get('forward_speed_mps', .1) if forward and policy else 0.
        if forward and forward_speed not in (.1, .3, .4):
            raise ValueError('前进速度档不符')
        required = 5. if forward else 2.
        if not policy or policy[-1]['time'] - policy[0]['time'] < required - .1:
            raise ValueError(f'策略运行不足完整{required:g}秒')
        for row in policy:
            expected = np.zeros(3)
            if forward:
                elapsed = row['feedback']['test_elapsed'] - 8.
                if 0 <= elapsed < 2:
                    ratio = min(elapsed / .3, (2 - elapsed) / .3, 1.)
                    expected[0] = forward_speed * ratio * ratio * (3 - 2 * ratio)
            # 500Hz行记录缓存的50Hz速度；允许一个20ms计划周期及少量调度偏差。
            if not np.allclose(row['velocity_command'], expected, atol=forward_speed * .15 if forward else 1e-7):
                raise ValueError('速度指令不符合固定时间曲线')
        if forward:
            moving = [r for r in policy if r['feedback']['test_elapsed'] >= 8 and r['feedback']['test_elapsed'] < 10]
            stopped = [r for r in policy if r['feedback']['test_elapsed'] >= 10.1]
            if not moving or max(r['velocity_command'][0] for r in moving) < forward_speed * .95:
                raise ValueError('缺少有效前进指令')
            if not stopped or stopped[-1]['time'] - stopped[0]['time'] < 1.7 or not all(np.allclose(r['velocity_command'], 0) for r in stopped):
                raise ValueError('缺少完整归零观察阶段')
        for row in policy:
            cap = row.get('feedback', {}).get('policy_pd_limit_nm')
            if cap is not None:
                pd = (np.asarray(row['kp_sdk']) * (np.asarray(row['target_pos_sdk']) - np.asarray(row['joint_pos_sdk']))
                      - np.asarray(row['kd_sdk']) * np.asarray(row['joint_vel_sdk']))
                if not np.isfinite(pd).all() or not np.isfinite(cap) or cap <= 0 or np.max(np.abs(pd)) > cap + 1e-4:
                    raise ValueError('策略下发PD不符合声明限幅')
            blend = row.get('feedback', {}).get('policy_entry_blend_seconds', 0.)
            ratio = np.clip((row['feedback']['test_elapsed'] - 7.) / blend, 0, 1) if blend else 1.
            ratio = ratio * ratio * (3 - 2 * ratio)
            expected_kp = (_array(config.data['fsm']['stand_kp'], 'stand_kp')[config.sdk_to_policy] * (1-ratio)
                           + config.stiffness[config.sdk_to_policy] * ratio)
            expected_kd = (_array(config.data['fsm']['stand_kd'], 'stand_kd')[config.sdk_to_policy] * (1-ratio)
                           + config.damping[config.sdk_to_policy] * ratio)
            if not np.allclose(row['kp_sdk'], expected_kp) or not np.allclose(row['kd_sdk'], expected_kd):
                raise ValueError('策略增益或接管增益曲线不符')
        active = [r for r in rows if r.get('mode') in ('policy', 'fix_stand')]
        if active[-1]['time'] - active[0]['time'] < (18.8 if forward else 15.8):
            raise ValueError('完整控制时长记录不足；需核对接管计时起点')
        final = rows[-1]
        state = type('RecordedState', (), {'timestamp': final['time'], 'joint_pos_sdk': final['joint_pos_sdk'],
                    'joint_vel_sdk': final['joint_vel_sdk'], 'quaternion_wxyz': final['quaternion_wxyz']})()
        check_prone(config, state, final['time'])
        return dict(result, result='forward_policy_trial_log_passed' if forward else 'zero_velocity_policy_trial_passed',
                    policy_seconds=policy[-1]['time'] - policy[0]['time'],
                    inference_count=policy[-1]['feedback']['policy_inference_count'],
                    **({'physical_translation_and_stop_confirmed': False} if forward else {}))
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        return dict(result, reason=str(error))

def assess_keyboard(rows, config):
    result={'result':'inconclusive','physical_translation_and_stop_confirmed':False}
    fault=next((r.get('feedback',{}).get('policy_fault') for r in rows if r.get('feedback',{}).get('policy_fault')),None)
    if fault is not None: result['policy_fault']=fault
    try:
        profile=next((r.get('feedback',{}).get('stand_profile') for r in rows if r.get('mode')=='policy'),None)
        continuous = profile == 'keyboard_continuous'
        allowed_reasons = ('', '主终端Z请求停止', '主终端Ctrl+C请求停止', '第二终端请求停止', '收到停止信号', '测试时长结束') if continuous else ('', '测试时长结束')
        if any(r.get('reason') not in allowed_reasons for r in rows):
            raise ValueError('键盘控制发生故障或非预期停止')
        profiles={'keyboard23':(14.,2.,.3,23.), 'keyboard29':(20.,4.,.4,29.), 'keyboard35':(26.,10.,.4,35.)}
        if continuous:
            window_end, budget, speed, duration = float('inf'), float('inf'), .4, float('inf')
        elif profile in profiles:
            window_end,budget,speed,duration=profiles[profile]
        else:
            raise ValueError('未知键盘时长档位')
        policy=[r for r in rows if r.get('mode')=='policy']
        if not policy or (not continuous and policy[-1]['time']-policy[0]['time']<window_end-5.2):
            raise ValueError('缺少完整键盘策略阶段')
        axis=policy[0]['feedback'].get('keyboard_axis','forward')
        if (profile == 'keyboard35' or continuous) and axis != 'all':
            raise ValueError('持续/10秒运动仅允许统一键盘')
        for row in policy:
            f=row['feedback']
            expected_budget = None if continuous else budget
            expected_window = None if continuous else window_end
            if (f.get('stand_profile') != profile or f.get('keyboard_axis','forward') != axis
                    or f.get('keyboard_motion_budget_seconds',expected_budget) != expected_budget
                    or f.get('keyboard_window_end_seconds',expected_window) != expected_window
                    or (continuous and f.get('keyboard_continuous') is not True)):
                raise ValueError('键盘时长/运动预算标记不一致')
        feedback_modes = {(r.get('feedback', {}).get('velocity_feedback') or {}).get('mode', 'off') for r in policy}
        if len(feedback_modes) != 1 or not feedback_modes <= {'off', 'shadow', 'pi'}:
            raise ValueError('速度反馈模式不一致')
        pi_mode = feedback_modes == {'pi'}
        if pi_mode and (not continuous or axis != 'all'):
            raise ValueError('PI候选仅支持all持续控制')
        result['velocity_feedback_mode'] = next(iter(feedback_modes))
        index={'forward':0,'backward':0,'lateral':1,'yaw':2,'all':None}[axis]
        keys={'forward':('w',),'backward':('s',),'lateral':('a','d'),'yaw':('q','e'),'all':('q','w','e','a','s','d')}[axis]
        profiles = {r['feedback'].get('speed_profile','legacy') for r in policy}
        if len(profiles) != 1 or not profiles <= {'legacy','uniform'}:
            raise ValueError('速度配置标记无效或不一致')
        uniform = profiles == {'uniform'}
        if uniform and (axis != 'all' or not continuous):
            raise ValueError('uniform只支持all持续控制')
        yaw_limit=policy[0]['feedback'].get('yaw_speed_rad_s',.3)
        if yaw_limit not in (.3,.4,.5): raise ValueError('未知转向限幅')
        lateral_limit=policy[0]['feedback'].get('lateral_speed_mps', .22)
        if not any(abs(lateral_limit-value)<1e-6 for value in (.22, .3)): raise ValueError('未知侧移限幅')
        limit={'forward':speed,'backward':.3,'lateral':lateral_limit,'yaw':yaw_limit,'all':.3}[axis]
        turn_forward=policy[0]['feedback'].get('turn_forward_speed_mps',0.)
        if turn_forward not in ((.5,) if uniform else (0.,.3)) or (turn_forward and axis not in ('yaw','all')):
            raise ValueError('未知边走边转配置')
        left_speed = policy[0]['feedback'].get('left_speed_mps', .3)
        if left_speed not in (.3,.4) or any(r['feedback'].get('left_speed_mps',.3) != left_speed for r in policy):
            raise ValueError('A左移速度标记无效或不一致')
        walk_speed = policy[0]['feedback'].get('forward_speed_mps',float('nan'))
        if axis == 'all' and (not any(np.isclose(walk_speed,x,atol=1e-6,rtol=0) for x in (.3,.5))
                or any(not np.isclose(r['feedback'].get('forward_speed_mps',float('nan')),walk_speed,atol=1e-6,rtol=0) for r in policy)):
            raise ValueError('统一键盘W速度标记无效或不一致')
        if uniform and (walk_speed != .5 or left_speed != .4 or yaw_limit != .5 or abs(lateral_limit-.3)>1e-6):
            raise ValueError('uniform速度配置不符')
        if not uniform and axis == 'all' and (yaw_limit != .3 or abs(lateral_limit-.3)>1e-6 or turn_forward != .3):
            raise ValueError('统一键盘限幅配置不符')
        action_modes = {(r['feedback'].get('policy_action_processing') or {}).get('mode','reject') for r in policy}
        if len(action_modes) != 1 or not action_modes <= {'reject','limit'}:
            raise ValueError('动作步长处理模式不一致')
        if axis == 'all' and (walk_speed > .3+1e-6 or left_speed > .3) and action_modes != {'limit'}:
            raise ValueError('提高前进速度必须记录动作步长限幅')
        if action_modes == {'limit'}:
            ceiling = min(float(config.safety['max_action_delta']), 8.*config.control_dt/float(config.data['actions']['scale']))
            for row in policy:
                processing = row['feedback']['policy_action_processing']
                step_limit = processing['step_limit']
                previous = _array(processing['previous_action'],'previous_action')
                applied = _array(processing['applied_action'],'applied_action')
                raw = _array(processing['raw_action'],'raw_action')
                if (not 0 < step_limit <= ceiling+1e-7 or np.max(np.abs(applied-previous)) > step_limit+1e-6
                        or np.max(np.abs(raw)) > config.safety['max_action']
                        or np.max(np.abs(applied)) > config.safety['max_action']):
                    raise ValueError('动作步长限幅记录越界')
        def moving_command(row):
            v = np.asarray(row['velocity_command'])
            return bool(np.any(np.abs(v)>1e-4)) if axis == 'all' else abs(v[index])>1e-4
        moving=[r for r in policy if moving_command(r)]
        if not moving: raise ValueError('没有键盘运动记录')
        for r in policy:
            v=np.asarray(r['velocity_command'])
            if v.shape != (3,) or not np.isfinite(v).all():
                raise ValueError('键盘速度必须为有限三维向量')
            if pi_mode:
                observation = r['feedback']['velocity_feedback']
                candidate = np.asarray(observation.get('candidate'), dtype=float)
                target = np.asarray(observation.get('candidate_target'), dtype=float)
                valid = (observation.get('valid') is True and observation.get('frame') in ('body', 'world')
                         and observation.get('age_seconds', float('inf')) <= .1
                         and candidate.shape == (3,) and target.shape == (3,)
                         and np.isfinite(candidate).all() and np.isfinite(target).all()
                         and np.all(abs(v) <= np.asarray([.6,.5,.6] if uniform else config.data['command_limits'])+1e-5)
                         and np.allclose(v, candidate, atol=1e-6, rtol=0)
                         and np.all(abs(target) <= np.asarray([.5,.4,.5] if uniform else [.2]*3)+1e-6)
                         and (np.any(target) or not np.any(v)))
            elif axis == 'all':
                translation = v[2] == 0 and (v[0] == 0 or v[1] == 0)
                turning = v[1] == 0 and np.isclose(v[0], abs(v[2]), atol=1e-6, rtol=0)
                valid = ((translation and -(walk_speed if uniform else .3)-1e-5 <= v[0] <= walk_speed+1e-5 and -(left_speed if uniform else .3)-1e-5 <= v[1] <= left_speed+1e-5)
                         or (turning and np.max(np.abs(v)) <= (.5 if uniform else .3)+1e-5))
            else:
                expected=np.zeros(3);expected[index]=v[index]
                if turn_forward: expected[0]=turn_forward*abs(v[index])/yaw_limit
                valid = (abs(v[index])<=limit+1e-5 and not (axis=='forward' and v[0]<0)
                         and not (axis=='backward' and v[0]>0) and np.allclose(v,expected,atol=1e-6,rtol=0))
            if not valid:
                raise ValueError('键盘速度超出受限运动范围')
            if moving_command(r) and not 8<=r['feedback']['test_elapsed']<window_end+.05:
                raise ValueError('运动出现在输入窗口之外')
            if r['feedback']['keyboard_forward_seconds']>budget+.06:
                raise ValueError('累计运动超出预算加调度余量')
        events={r['feedback']['keyboard_event']['sequence']:r['feedback']['keyboard_event'] for r in policy if r['feedback'].get('keyboard_event') and r['feedback']['keyboard_event'].get('accepted',True)}
        if not any(e['key'] in keys for e in events.values()): raise ValueError('缺少方向键输入')
        if not continuous and not any(e['key'] in ('x',' ') and e['elapsed']>moving[0]['feedback']['test_elapsed'] for e in events.values()):
            raise ValueError('没有运动后的X/空格归零记录')
        stops=[e for e in events.values() if e['key'] in ('x',' ')]
        active_x = any(any(0<=e['elapsed']-r['feedback']['test_elapsed']<=.1 and moving_command(r) for r in policy)
                and any(0<=r['feedback']['test_elapsed']-e['elapsed']<=.06 and not np.any(r['velocity_command']) for r in policy) for e in stops)
        if not continuous and not active_x:
            raise ValueError('未记录到X/空格从运动切换归零')
        result['command_checks_passed'] = True
        if continuous:
            normal_exit = next((r['feedback'].get('keyboard_normal_exit') for r in rows if r.get('feedback',{}).get('keyboard_normal_exit')), None)
            if rows[-1].get('reason') == '测试时长结束':
                if normal_exit is None or rows[-1]['feedback']['test_elapsed'] < normal_exit['duration']:
                    raise ValueError('缺少C受控降低完成记录')
            if rows[-1].get('mode') != 'damping' or rows[-1].get('reason') not in allowed_reasons[1:]:
                raise ValueError('持续控制缺少主动停止及阻尼退出记录')
            return dict(result, result='keyboard_continuous_normal_exit_log_passed' if rows[-1].get('reason') == '测试时长结束' else 'keyboard_continuous_manual_stop_log_passed', keyboard_axis=axis,
                        active_x_log_confirmed=active_x,
                        forward_seconds=max(r['feedback']['keyboard_forward_seconds'] for r in policy))
        active=[r for r in rows if r.get('mode') in ('fix_stand','policy')]
        if active[-1]['time']-active[0]['time']<duration-.2: raise ValueError('完整键盘控制时长不足')
        final=rows[-1]
        state=type('RecordedState',(),{'timestamp':final['time'],'joint_pos_sdk':final['joint_pos_sdk'],'joint_vel_sdk':final['joint_vel_sdk'],'quaternion_wxyz':final['quaternion_wxyz']})()
        check_prone(config,state,final['time'])
        return dict(result,result='keyboard_unified_and_zero_log_passed' if axis=='all' else 'keyboard_forward_and_zero_log_passed' if axis=='forward' else 'keyboard_axis_and_zero_log_passed',keyboard_axis=axis,forward_seconds=max(r['feedback']['keyboard_forward_seconds'] for r in policy))
    except (ValueError,KeyError,TypeError,AttributeError) as error:
        return dict(result,reason=str(error))



@contextmanager
def bounded_control_gc():
    """有界验收进程在改原厂模式前清理循环垃圾，控制期间保留引用计数释放。"""
    enabled = gc.isenabled()
    gc.collect()
    if enabled:
        gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def serve(args, config):
    config_sha256 = None
    # 正常CLI总有配置路径；测试/程序化调用没有路径时不伪造版本标记。
    if getattr(args, 'config', None) is not None:
        with open(args.config, 'rb') as stream:
            config_sha256 = hashlib.sha256(stream.read()).hexdigest()
    with bounded_control_gc():
        return _serve(args, config, config_sha256)


def startup_lease(server, test, previous=None):
    """站立接管前取最新主终端租约；启动期间收到停止也不能开始主动控制。"""
    latest = previous
    while True:
        try:
            raw, address = server.recvfrom(4096)
        except BlockingIOError:
            break
        try:
            payload = json.loads(raw)
            if payload.get('type') == 'startup_heartbeat':
                stamp = float(payload['time'])
                if math.isfinite(stamp) and (latest is None or stamp > latest):
                    latest = stamp
            elif payload.get('type') == 'stop':
                test.stop(time.monotonic(), '启动期间请求停止')
                if address:
                    server.sendto(json.dumps({'stopped': True, 'dds_submitted': False, 'reason': test.reason}).encode(), address)
        except (ValueError, TypeError, KeyError):
            test.stop(time.monotonic(), '无效启动租约')
        except OSError:
            pass
    if latest is None or not 0 <= time.monotonic() - latest < COMMAND_TIMEOUT:
        test.stop(time.monotonic(), '启动租约缺失或超时')
    return latest


def attach_velocity_feedback(test, args, config_sha256):
    if test.config.data.get("format_version") == 2 and (getattr(args, 'gait_restart', False) or getattr(args, 'velocity_feedback', 'off') == 'pi'):
        raise ValueError("HIM初次部署未验证PI或起步恢复叠加")
    if getattr(args, 'gait_restart', False):
        from deploy.common.go2_gait_restart import GaitRestart
        test.gait_restart = GaitRestart()
    if getattr(args, 'velocity_feedback', 'off') == 'off':
        return
    from deploy.deploy_real.go2_velocity_feedback import VelocityFeedback, validate_feedback_readiness
    test.velocity_feedback = VelocityFeedback(args.velocity_feedback, test.controller.config.data['command_limits'],
        getattr(args, 'velocity_frame', None), source=getattr(args,'velocity_source','sport'),
        config=test.config, speed_profile=test.speed_profile)
    if args.velocity_feedback == 'pi':
        if test.speed_profile == 'uniform':
            test.controller.config.data['command_limits'] = test.velocity_feedback.limits.tolist()
        test.velocity_readiness = validate_feedback_readiness(args.velocity_readiness,
                                                              test.policy_sha256, config_sha256,
            source=test.velocity_feedback.source_name,
            estimator_sha256=getattr(test.velocity_feedback.source,'fingerprint',None),
            speed_profile=test.speed_profile if test.speed_profile == 'uniform' else None,
            profile_sha256=test.velocity_feedback.profile_sha256 if test.speed_profile == 'uniform' else None)


def _serve(args, config, config_sha256):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    path = default_socket(0)
    standing = getattr(args, 'start_standing', False)
    test = PolicyCheck(config, args.forward_check or args.policy_check, forward=bool(args.forward_check), forward_speed=getattr(args, 'forward_speed', .1), keyboard=getattr(args, 'keyboard', False), keyboard_axis=getattr(args,'keyboard_axis','forward'), yaw_speed=getattr(args,'yaw_speed',.3), turn_forward_speed=getattr(args,'turn_forward_speed',0.), keyboard_motion_seconds=getattr(args,'keyboard_motion_seconds',None), continuous=getattr(args,'continuous',False), walk_speed=getattr(args,'walk_speed',.3), action_step_mode=getattr(args,'action_step_mode',None), left_speed=getattr(args,'left_speed',.3), speed_profile=getattr(args,'speed_profile','legacy')) if getattr(args, 'policy_check', None) or getattr(args, 'forward_check', None) else StandCheck(config, standing) if getattr(args, 'stand_check', False) else ProneTest(
        config, args.test_joint, args.test_amplitude, args.duration, args.motor_test,
        response=getattr(args, 'response_test', False), joint_check=getattr(args, 'joint_check', False))
    attach_velocity_feedback(test, args, config_sha256)
    publish_dt = 1.0 / getattr(args, 'publish_hz', 500)
    transport = recorder = None
    released = False
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as server:
        server.bind(path)
        os.chmod(path, 0o600)
        server.setblocking(False)
        try:
            ChannelFactoryInitialize(0, args.network)
            reader = VerifiedTransport(config, publish=False)
            if isinstance(test, PolicyCheck) and test.velocity_feedback is not None:
                test.velocity_feedback.source.start()
                if test.velocity_feedback.mode == 'pi':
                    deadline = time.monotonic() + 3.
                    while True:
                        try:
                            if test.velocity_feedback.source_name == 'kinematic':
                                test.velocity_feedback.source.observe(reader.snapshot(), time.monotonic())
                            test.velocity_feedback.source.snapshot(time.monotonic())
                            break
                        except ValueError:
                            if time.monotonic() >= deadline:
                                raise ValueError('接管前未收到有效机身速度；原厂模式未释放，未发LowCmd')
                            time.sleep(.02)
            deadline = time.monotonic() + 5
            while reader.snapshot() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError('等待LowState超时')
                time.sleep(0.01)
            # 原厂站起可能尚在插值：等待连续一秒站稳，等待期间不改模式或发命令。
            if standing:
                print('等待原厂连续一秒站稳；此时不切换模式、不发送LowCmd。', flush=True)
                deadline = time.monotonic() + 15
                stable_since = None
                while True:
                    now = time.monotonic()
                    try:
                        check_standing(config, reader.snapshot(), now)
                        stable_since = now if stable_since is None else stable_since
                    except ValueError:
                        stable_since = None
                    if stable_since is not None and now - stable_since >= 1:
                        break
                    if now >= deadline:
                        raise ValueError('15秒内原厂未连续站稳；原厂模式未释放，未下发LowCmd')
                    time.sleep(config.control_dt)
            else:
                until = time.monotonic() + 1
                while time.monotonic() < until:
                    check_prone(config, reader.snapshot(), time.monotonic())
                    time.sleep(config.control_dt)
            client = mode_client()
            if standing:
                if read_mode(client) != 'normal':
                    raise ValueError('站立接管要求原厂normal正在保持站姿')
                # 原厂仍保持站姿时完成DDS发布器和日志初始化；此时不写任何LowCmd。
                transport = reader
                transport.prepare_factory_handoff()
                heartbeat = startup_lease(server, test)
                if test.reason:
                    print(f'站立接管取消: {test.reason}；原厂模式未释放，未发LowCmd。', flush=True)
                    return
            recording = dict(config.data['recording'])
            recording['directory'] += '/commission'
            recorder = Go2Recorder(recording, LEGGED_GYM_ROOT_DIR)
            if standing:
                # 起点必须在原厂仍承重时严格验证；释放后不能再要求其原厂静止。
                heartbeat = startup_lease(server, test, heartbeat)
                if test.reason:
                    print(f'站立接管取消: {test.reason}；原厂未释放，未发LowCmd。', flush=True)
                    return
                test.arm(time.monotonic(), transport.snapshot())
            previous = release_mode(client)
            released = True
            print(f'原厂模式已释放（先前{previous!r}）；结束后需显式恢复normal。', flush=True)
            heartbeat = heartbeat if standing else None
            if standing:
                # 第一条保持命令紧接释放确认；启动租约最多100ms，需主终端及时接续。
                heartbeat = startup_lease(server, test, heartbeat)
                if not test.reason:
                    test.begin_handoff(time.monotonic())
                transport.guard_command_source = True
                transport.send(test.output(time.monotonic(), transport.snapshot()))
            else:
                check_prone(config, reader.snapshot(), time.monotonic())
                # 内置探针拒绝其他LowCmd发布者；在释放服务后才创建发布器。
                transport = VerifiedTransport(config, publish=True)
            signal.signal(signal.SIGINT, lambda *_: test.stop(time.monotonic(), '收到停止信号'))
            signal.signal(signal.SIGTERM, lambda *_: test.stop(time.monotonic(), '收到停止信号'))
            idle_deadline = time.monotonic() + 5
            next_step = time.monotonic()
            keyboard_notice = None
            gait_notice_sequence = 0
            while True:
                state = transport.snapshot()
                replies = []
                for _ in range(32):
                    try:
                        raw, address = server.recvfrom(4096)
                    except BlockingIOError:
                        break
                    try:
                        payload = json.loads(raw)
                        kind = payload['type']
                        now = time.monotonic()
                        if kind == 'status':
                            server.sendto(json.dumps({'ready': True, 'stopped': bool(test.reason), 'pid': os.getpid(),
                                                      'kind': 'stand_check' if getattr(args, 'stand_check', False) else 'prone_motor' if args.motor_test else 'prone_damping'}).encode(), address)
                        elif kind == 'stop':
                            source = str(payload.get('source','external'))[:64]
                            reason = {'keyboard_z':'主终端Z请求停止','ctrl_c':'主终端Ctrl+C请求停止',
                                      'parent_cleanup':'主终端结束请求停止','parent_error':'主终端异常退出请求停止'}.get(source,'第二终端请求停止')
                            if not test.reason:
                                test.stop_request = {'source':source, 'keys':str(payload.get('keys',[]))[:128],
                                                     'error':str(payload.get('error',''))[:256]}
                            test.stop(now, reason)
                            replies.append(address)
                        elif kind in ('arm', 'heartbeat', 'startup_heartbeat'):
                            timestamp = float(payload['time'])
                            if not math.isfinite(timestamp) or not 0 <= now - timestamp <= COMMAND_TIMEOUT:
                                raise ValueError('主程序心跳过期')
                            if heartbeat is not None and timestamp <= heartbeat:
                                raise ValueError('主程序心跳时间未递增')
                            heartbeat = timestamp
                            keys = payload.get('keys', [])
                            if not isinstance(keys,list) or len(keys)>32 or not all(isinstance(k,str) and len(k)==1 for k in keys):
                                raise ValueError('键盘消息格式无效')
                            if isinstance(test,PolicyCheck):
                                test.feed_keys(keys,now)
                            if kind == 'arm':
                                if not standing or test.started is None:
                                    test.arm(now, state)
                        else:
                            raise ValueError('未知请求')
                    except (ValueError, KeyError, TypeError) as error:
                        test.stop(time.monotonic(), str(error))
                    except OSError:
                        pass
                now = time.monotonic()
                if getattr(transport, 'foreign_command', False):
                    test.stop(now, '站立接管检测到不匹配的LowCmd或错误CRC')
                if test.started is not None and not test.reason and now - heartbeat >= COMMAND_TIMEOUT:
                    test.stop(now, '主程序心跳超时')
                command = test.output(now, state)
                if isinstance(test, PolicyCheck) and test.gait_restart is not None:
                    event = test.gait_restart.event
                    if event is not None and event['sequence'] != gait_notice_sequence:
                        gait_notice_sequence = event['sequence']
                        notice = {
                            'stand_begin': '左移步态持续静止，正在回站姿；X取消继续左移，C正常退出，Z急停。',
                            'stand_ready': '站姿稳定，正在重新接入策略；按住方向键继续。' if test.normal_exit is None else '站姿稳定，继续正常降低退出。',
                            'left_blocked': '本次左移仍未形成步态，已归零；X后可重试，或C正常退出。',
                        }.get(event['event'])
                        if notice:
                            print(notice, flush=True)
                if isinstance(test,PolicyCheck) and test.keyboard and test.started is not None:
                    e = now-test.started
                    notice = f'按住{"/".join(test.keyboard_keys).upper()}控制{test.keyboard_axis}，X/空格归零；累计运动最多{test.keyboard_budget:g}秒。' if 8<=e<test.keyboard_window_end else '即将开放键盘窗口，请准备所选方向键和X。' if 7<=e<8 else '输入窗口已结束，速度归零，随后自动降低。' if e>=test.keyboard_window_end else None
                    if test.normal_exit is not None:
                        notice='C正常退出：速度归零后受控降低，请等待流程结束；Z仍可紧急终止。'
                    elif test.continuous and e>=8:
                        notice='持续键盘控制：Q/W/E/A/S/D，X/空格停走，C受控降低退出，Z/Ctrl+C紧急阻尼；无累计运动预算。'
                    if 8<=e<test.keyboard_window_end and test.keyboard_used>=test.keyboard_budget:
                        notice='累计运动预算已耗尽，方向键不再接受；速度归零，随后自动降低。'
                    if notice is not None and notice!=keyboard_notice:
                        print(notice,flush=True); keyboard_notice=notice
                if command is not None:
                    transport.send(command)
                    if state is not None:
                        feedback = dict(getattr(state, 'feedback', None) or {})
                        if getattr(test,'stop_request',None) is not None:
                            feedback['stop_request'] = test.stop_request
                        feedback['control_config_sha256'] = config_sha256
                        feedback['test_phase'] = test.phase(now)[0]
                        feedback['test_elapsed'] = None if test.started is None else now - test.started
                        feedback['test_joint_sdk'] = test.joint
                        feedback['range_transition'] = test.recover_range
                        if isinstance(test, StandCheck):
                            feedback['handoff_delay_seconds'] = test.handoff_delay_seconds
                            feedback['stand_profile'] = (('keyboard_continuous' if test.continuous else f'keyboard{int(test.duration)}') if test.keyboard else 'forward19' if test.forward else 'policy16') if isinstance(test, PolicyCheck) else 'standing14' if standing else 'prone17'
                            if isinstance(test, PolicyCheck):
                                if test.gait_restart is not None:
                                    feedback['gait_restart'] = test.gait_restart.audit()
                                if test.velocity_feedback is not None:
                                    feedback['velocity_feedback'] = dict(test.velocity_feedback.audit)
                                    feedback['velocity_profile_sha256'] = test.velocity_feedback.profile_sha256
                                    if hasattr(test, 'velocity_readiness'):
                                        feedback['velocity_readiness'] = test.velocity_readiness
                                feedback['policy_inference_count'] = test.inference_count
                                feedback['policy_sha256'] = test.policy_sha256
                                feedback['policy_config_sha256'] = config_sha256
                                feedback['policy_target_processing'] = test.controller.last_target_processing
                                feedback['policy_action_processing'] = test.controller.last_action_processing
                                if test.controller.last_policy_fault is not None:
                                    feedback['policy_fault'] = test.controller.last_policy_fault
                                feedback['policy_entry_blend_seconds'] = test.entry_blend_seconds
                                feedback['policy_pd_limit_nm'] = test.policy_pd_limit
                                feedback['policy_feedback_limit_nm'] = test.torque_limit
                                feedback['settle_tracking_tolerance_rad'] = test.settle_tracking_tolerance
                                feedback['speed_profile'] = test.speed_profile
                                feedback['left_speed_mps'] = test.left_speed
                                feedback['effective_command_limits'] = list(test.controller.config.data['command_limits'])
                                feedback['forward_speed_mps'] = test.walk_speed if test.keyboard_axis == 'all' else test.forward_speed if test.forward else 0.
                                feedback['policy_joint_velocity_limit_rad_s'] = test.policy_velocity_limit
                                feedback['policy_joint_velocity_limits_sdk_rad_s'] = test.policy_velocity_limits.tolist()
                                feedback['policy_limits_version'] = 'per_joint_v1'
                                if test.keyboard:
                                    feedback['keyboard_event'] = test.keyboard_event
                                    feedback['keyboard_forward_seconds'] = test.keyboard_used
                                    feedback['keyboard_motion_budget_seconds'] = None if test.continuous else test.keyboard_budget
                                    feedback['keyboard_window_end_seconds'] = None if test.continuous else test.keyboard_window_end
                                    feedback['keyboard_normal_exit'] = test.normal_exit
                                    feedback['keyboard_continuous'] = test.continuous
                                    feedback['keyboard_timeout_seconds'] = test.keyboard_timeout
                                    feedback['keyboard_axis'] = test.keyboard_axis
                                    feedback['lateral_speed_mps'] = float(test.keyboard_input.limits[1])
                                    feedback['yaw_speed_rad_s'] = test.yaw_speed
                                    feedback['turn_forward_speed_mps'] = test.turn_forward_speed
                        recorder.write(now, state, command, test.controller.command if isinstance(test, PolicyCheck) else np.zeros(3), feedback=feedback)
                for address in replies:
                    try:
                        server.sendto(json.dumps({'stopped': True, 'reason': test.reason}).encode(), address)
                    except OSError:
                        pass
                if test.reason and now - test.stopped >= 1:
                    print(f'测试停止: {test.reason}；已发送一秒阻尼。', flush=True)
                    if test.reason not in ('测试时长结束', '第二终端请求停止', '收到停止信号',
                                           '主终端Z请求停止','主终端Ctrl+C请求停止','主终端结束请求停止'):
                        raise RuntimeError(f'验收未通过: {test.reason}')
                    break
                if test.started is None and not test.reason and now >= idle_deadline:
                    break
                next_step += publish_dt
                if next_step < time.monotonic():
                    next_step = time.monotonic() + publish_dt
                time.sleep(max(0, next_step - time.monotonic()))
        finally:
            try:
                if released and transport is not None and test.started is not None and (
                        test.stopped is None or time.monotonic() - test.stopped < 1):
                    test.stop(time.monotonic(), '验收进程异常退出')
                    until = time.monotonic() + 1
                    while time.monotonic() < until:
                        transport.send(test.output(time.monotonic(), transport.snapshot()))
                        time.sleep(publish_dt)
            finally:
                if isinstance(test, PolicyCheck) and test.velocity_feedback is not None:
                    test.velocity_feedback.source.close()
                if recorder:
                    recorder.close()
                os.unlink(path)


def run(args, config):
    path = default_socket(0)
    if os.path.lexists(path):
        raise RuntimeError(f'停止接口已存在: {path}；先检查旧进程，不能覆盖')
    command = [sys.executable, '-m', 'deploy.deploy_real.go2_commission', args.network,
               '--serve', '--confirm-standing' if args.start_standing else '--confirm-prone', '--release-mode', '--config', args.config,
               '--duration', str(args.duration), '--test-joint', args.test_joint,
               '--publish-hz', str(args.publish_hz),
               '--test-amplitude', str(args.test_amplitude),
               '--forward-check' if args.forward_check else '--policy-check' if args.policy_check else '--stand-check' if args.stand_check else '--joint-check' if args.joint_check else '--response-test' if args.response_test else ('--motor-test' if args.motor_test else '--damping-test')]
    if args.policy_check or args.forward_check:
        command.extend([args.forward_check or args.policy_check, '--allow-policy-movement',
                        '--action-step-mode', resolve_action_step_mode(getattr(args, 'action_step_mode', None), bool(args.forward_check))])
        if args.forward_check:
            command.extend(['--forward-speed', str(args.forward_speed)])
    if args.stand_check or args.policy_check or args.forward_check:
        command.append('--allow-body-movement')
    if args.keyboard:
        command.extend(['--keyboard','--keyboard-axis',args.keyboard_axis,'--yaw-speed',str(args.yaw_speed),'--turn-forward-speed',str(args.turn_forward_speed)])
        command.extend(['--walk-speed',str(getattr(args,'walk_speed',.3))])
        if args.keyboard_motion_seconds is not None:
            command.extend(['--keyboard-motion-seconds',str(args.keyboard_motion_seconds)])
    if getattr(args, 'speed_profile', 'legacy') != 'legacy':
        command.extend(['--speed-profile', args.speed_profile])
    if getattr(args, 'left_speed', .3) != .3:
        command.extend(['--left-speed', str(args.left_speed)])
    if getattr(args, 'gait_restart', False):
        command.append('--gait-restart')
    if getattr(args, 'velocity_feedback', 'off') != 'off':
        command.extend(['--velocity-feedback', args.velocity_feedback, '--velocity-source', getattr(args,'velocity_source','sport')])
        if getattr(args, 'velocity_frame', None):
            command.extend(['--velocity-frame', args.velocity_frame])
        if getattr(args, 'velocity_readiness', None):
            command.extend(['--velocity-readiness', args.velocity_readiness])
    if getattr(args, 'continuous', False):
        command.append('--continuous')
    if args.start_standing:
        command.append('--start-standing')
    process = subprocess.Popen(command, cwd=LEGGED_GYM_ROOT_DIR, start_new_session=True)
    keys = TerminalKeys()
    pressed = []
    try:
        deadline = time.monotonic() + 20
        while True:
            if process.poll() is not None:
                raise RuntimeError('真机验收进程启动失败；查看上方错误，原厂模式可能需要恢复')
            try:
                if args.start_standing:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as startup:
                        startup.sendto(json.dumps({'type': 'startup_heartbeat', 'time': time.monotonic()}).encode(), path)
                if request(path, {'type': 'status'}, timeout=0.02 if args.start_standing else 0.1).get('ready'):
                    break
            except (OSError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError('等待真机验收进程超时')
            time.sleep(0.02)
        keys.start()
        if args.turn_forward_speed:
            print('本轮为边走边转：Q/E同时产生向前移动，X归零两个轴。',flush=True)
        print((f'键盘验收轴：{args.keyboard_axis}；听到窗口提示后输入，X/空格归零；C受控降低，Z/Ctrl+C紧急阻尼。' if args.keyboard else 'X、空格、Z或Ctrl+C停止。')+' 第二终端：python -m deploy.deploy_real.go2_watchdog --stop --domain-id 0',flush=True)
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
            sender.sendto(json.dumps({'type': 'arm', 'time': time.monotonic()}).encode(), path)
            while process.poll() is None:
                pressed = keys.poll()
                if any(key in (('z',) if args.keyboard else ('x',' ','z')) for key in pressed):
                    source = 'keyboard_z' if 'z' in pressed else 'parent_cleanup'
                    request(path, {'type': 'stop', 'source':source, 'keys':pressed})
                    break
                try:
                    sender.sendto(json.dumps({'type': 'heartbeat', 'time': time.monotonic(), 'keys': pressed if args.keyboard else []}).encode(), path)
                except FileNotFoundError:
                    break
                time.sleep(config.control_dt)
        process.wait(timeout=4)
        if process.returncode:
            raise RuntimeError('真机验收进程异常退出；不能认定测试通过')
    finally:
        error = sys.exc_info()[1]
        try:
            keys.close()
        finally:
            if process.poll() is None:
                source = 'ctrl_c' if isinstance(error,KeyboardInterrupt) else 'parent_error' if error is not None else 'parent_cleanup'
                try:
                    request(path, {'type': 'stop', 'source':source, 'keys':pressed,
                                   'error':'' if error is None else f'{type(error).__name__}: {error}'}, timeout=0.5)
                except OSError:
                    pass
                # 不强制结束独立进程；心跳消失后它自行发送阻尼。
                process.wait(timeout=5)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('network', nargs='?')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='离线检查，不初始化DDS')
    mode.add_argument('--analyze', metavar='JSONL', help='只分析响应测试日志，不初始化DDS')
    mode.add_argument('--damping-test', action='store_true')
    mode.add_argument('--motor-test', action='store_true')
    mode.add_argument('--response-test', action='store_true', help='3秒基线/正负目标响应测试，Kp=20，仅选定髋关节')
    mode.add_argument('--joint-check', action='store_true', help='逐关节验收：固定0.02弧度、Kp=40、3秒')
    mode.add_argument('--stand-check', action='store_true', help='17秒趴卧归位/站立/保持/降低，不运行策略')
    mode.add_argument('--policy-check', metavar='READY_JSONL', help='已通过站立验收日志；固定2秒零速度策略，整轮16秒')
    mode.add_argument('--forward-check', metavar='ZERO_JSONL', help='通过的零速度策略日志；固定2秒前进及2秒归零观察')
    parser.add_argument('--keyboard',action='store_true',help='受限键盘运动/X归零；0.3档23秒/运动2秒，0.4档29秒/运动4秒')
    parser.add_argument('--turn-forward-speed',type=float,choices=(0.,.3,.5),default=0.,help='明确允许边走边转，0.30m/s前进；真机转向仅0.3rad/s')
    parser.add_argument('--yaw-speed',type=float,choices=(.3,.4,.5),default=.3,help='转向候选角速度，rad/s')
    parser.add_argument('--action-step-mode',choices=('reject','limit'),default=None,help='运动策略默认限幅；reject仅保留显式对照')
    parser.add_argument('--walk-speed',type=float,choices=(.3,.5),default=.3,help='all模式W纯前进速度；不改变S/A/D/Q/E')
    parser.add_argument('--continuous', action='store_true', help='all持续键盘控制，由Z/Ctrl+C退出；与运动预算互斥')
    parser.add_argument('--keyboard-motion-seconds', type=int, choices=(4,10), help='all统一键盘累计运动预算；10秒档35秒控制加1秒阻尼')
    parser.add_argument('--keyboard-axis',choices=('forward','backward','lateral','yaw','all'),default='forward',help='all统一Q/W/E/A/S/D；其余为单轴验收')
    parser.add_argument('--forward-speed', type=float, choices=(.1, .3, .4), default=.1, help='限时前进档；0.30m/s用于候选行走，0.10m/s保留历史对照')
    parser.add_argument('--allow-policy-movement', action='store_true', help='已知策略可能迈步/漂移或按固定指令前进')
    mode.add_argument('--restore-normal', action='store_true', help='无验收进程时显式恢复原厂normal模式')
    parser.add_argument('--confirm-prone', action='store_true', help='已现场确认机身趴在地面且周围无人')
    parser.add_argument('--start-standing', action='store_true', help='从原厂站姿接管，等待连续一秒站稳')
    parser.add_argument('--confirm-standing', action='store_true', help='已现场确认原厂站稳、周围无人')
    parser.add_argument('--release-mode', action='store_true', help='允许退出原厂运动模式')
    parser.add_argument('--allow-body-movement', action='store_true', help='已知站立验收会使整机站起和降低')
    parser.add_argument('--allow-factory-stand', action='store_true',
                        help='恢复normal可能自动站起，现场已准备App卧倒收尾')
    parser.add_argument('--serve', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--config', default=os.path.join(LEGGED_GYM_ROOT_DIR, 'deploy/deploy_real/configs/go2.yaml'))
    parser.add_argument('--duration', type=float)
    parser.add_argument('--test-joint', default='FL_hip_joint')
    parser.add_argument('--test-amplitude', type=float)
    parser.add_argument('--publish-hz', type=int, choices=(50, 500), default=500,
                        help='验收DDS发布频率，默认500Hz；50Hz仅供对照诊断')
    parser.add_argument('--velocity-feedback', choices=('off', 'shadow', 'pi'), default='off')
    parser.add_argument('--velocity-frame', choices=('body', 'world'), help='现场确认的机身速度坐标系')
    parser.add_argument('--velocity-readiness', help='PI前置接管后速度观测JSONL')
    parser.add_argument('--gait-restart', action='store_true', help='左移步态停滞时回站姿并重新接管一次')
    parser.add_argument('--left-speed', type=float, choices=(.3,.4), default=.3, help='仅改变all模式A左移指令')
    parser.add_argument('--speed-profile', choices=('legacy','uniform'), default='legacy')
    parser.add_argument('--velocity-source', choices=('sport','kinematic'), default='sport')
    args = parser.parse_args(argv)
    if args.velocity_source == 'kinematic':
        if args.velocity_feedback == 'off' or args.velocity_frame == 'world':
            parser.error('运动学速度需要shadow或pi，坐标系固定body')
        args.velocity_frame = 'body'
    uniform = args.speed_profile == 'uniform'
    if uniform and (not args.continuous or not args.keyboard or args.keyboard_axis != 'all' or args.walk_speed != .5 or args.left_speed != .4 or args.yaw_speed != .5 or args.turn_forward_speed != .5 or args.gait_restart):
        parser.error('uniform需要all持续控制和固定0.5/0.4/0.5速度配置，暂不叠加起步恢复')
    if args.left_speed != .3 and (not args.continuous or not args.keyboard or args.keyboard_axis != 'all' or (args.velocity_feedback == 'pi' and not uniform) or args.gait_restart):
        parser.error('A速度对照仅支持all，不能叠加PI或起步恢复')
    if args.gait_restart and (not args.continuous or not args.keyboard or args.keyboard_axis != 'all' or args.velocity_feedback == 'pi'):
        parser.error('--gait-restart仅支持all持续控制，暂不与PI叠加')
    if args.velocity_readiness and args.velocity_feedback != 'pi':
        parser.error('--velocity-readiness仅用于PI候选')
    if args.velocity_feedback != 'off' and (not args.keyboard or not args.forward_check or args.keyboard_axis != 'all'):
        parser.error('速度反馈仅用于all统一键盘策略')
    if args.velocity_feedback == 'pi' and (args.velocity_frame is None or args.velocity_readiness is None or (args.walk_speed != .3 and not uniform) or not args.continuous):
        parser.error('PI候选需要--velocity-frame和--velocity-readiness，使用持续模式；新速度使用uniform配置')
    if args.velocity_frame and args.velocity_feedback == 'off':
        parser.error('--velocity-frame需要启用速度观测或PI')
    try:
        if args.continuous and not args.keyboard:
            raise ValueError('--continuous需要--keyboard')
        if args.keyboard_motion_seconds is not None and not args.keyboard:
            raise ValueError('--keyboard-motion-seconds需要--keyboard')
        keyboard_duration = keyboard_timing(args.forward_speed, args.keyboard_axis, args.keyboard_motion_seconds, args.continuous)[2] if args.keyboard else None
    except ValueError as error:
        parser.error(str(error))
    if args.walk_speed != .3 and (not args.keyboard or args.keyboard_axis != 'all'):
        parser.error('--walk-speed 0.5仅允许all统一键盘')
    him_policy = False
    if args.policy_check or args.forward_check:
        from deploy.common.go2_policy import load_deploy_config
        him_policy = load_deploy_config(args.config).get('format_version') == 2
    try:
        args.action_step_mode = resolve_action_step_mode(args.action_step_mode, bool(args.forward_check) or him_policy)
    except ValueError as error:
        parser.error(str(error))
    if (args.walk_speed > .3 or args.left_speed > .3) and args.action_step_mode != 'limit':
        parser.error('提高--walk-speed必须同时指定--action-step-mode limit')
    policy_duration = (keyboard_duration if args.keyboard else 19 if args.forward_check else 16)
    if him_policy and not args.continuous:
        policy_duration += 14
    if args.duration is None:
        args.duration = policy_duration if args.policy_check or args.forward_check else (14 if args.start_standing else 17) if args.stand_check else 3 if args.response_test or args.joint_check else 2
    if args.test_amplitude is None:
        args.test_amplitude = 0.02 if args.joint_check else 0.005
    if args.keyboard and (not args.forward_check or args.forward_speed not in (.3, .4)):
        parser.error('--keyboard仅允许--forward-check并显式--forward-speed 0.3或0.4')
    if args.turn_forward_speed == .5 and not uniform:
        parser.error('0.5行进转向需要uniform速度配置')
    if args.turn_forward_speed and args.keyboard_axis not in ('yaw','all'):
        parser.error('边走边转需要yaw或all轴')
    if args.yaw_speed != .3 and args.keyboard_axis != 'yaw' and not uniform:
        parser.error('--yaw-speed 0.5仅允许yaw键盘档')
    if args.keyboard_axis != 'forward' and (not args.keyboard or args.forward_speed != .4):
        parser.error('后退/侧移/转向需要--keyboard --forward-speed 0.4')
    if not uniform and args.keyboard_axis == 'all' and (args.yaw_speed != .3 or args.turn_forward_speed != .3):
        parser.error('all固定平移0.30m/s和行进转向0.30rad/s，需--turn-forward-speed 0.3')
    if args.forward_speed == .4 and not args.keyboard:
        parser.error('0.40m/s仅用于受限键盘试跑')
    args.motor_test = args.motor_test or args.response_test or args.joint_check
    if args.start_standing and not (args.stand_check or args.policy_check or args.forward_check):
        parser.error('--start-standing只能与站立或限时策略检查一起使用')
    if not args.check and not args.analyze:
        if args.turn_forward_speed and args.yaw_speed != .3 and not uniform:
            parser.error('真实反馈回放显示0.4/0.5边走边转目标越界；真机仅允许0.3rad/s候选')
        if not args.network or args.network == 'lo' or not (args.confirm_standing if args.start_standing else args.confirm_prone):
            parser.error('真机操作必须指定实体网卡并确认相应的起始姿态')
        if not args.restore_normal and not args.release_mode:
            parser.error('真机测试必须显式--release-mode')
        if args.restore_normal and not args.allow_factory_stand:
            parser.error('恢复normal可能自动站起；现场准备App卧倒后显式--allow-factory-stand')
        if (args.policy_check or args.forward_check) and (not args.start_standing or not args.allow_policy_movement or not args.allow_body_movement or args.duration != policy_duration or args.publish_hz != 500):
            parser.error('策略试跑时长固定（零速16秒/前进19秒）且500Hz，需原厂站姿接管及显式动作确认')
        if args.stand_check and (not args.allow_body_movement or args.duration != (14 if args.start_standing else 17)):
            parser.error('站立验收时长固定，需显式--allow-body-movement')
        if not args.serve and not args.restore_normal and not sys.stdin.isatty():
            parser.error('真机测试必须使用前台交互式终端')
    return args


def main():
    args = parse_args()
    config = Go2RealConfig.load(os.path.abspath(args.config), LEGGED_GYM_ROOT_DIR)
    if not args.analyze:
        if args.policy_check or args.forward_check:
            checked_trial = PolicyCheck(config, args.forward_check or args.policy_check, forward=bool(args.forward_check), forward_speed=getattr(args, 'forward_speed', .1), keyboard=getattr(args, 'keyboard', False), keyboard_axis=getattr(args,'keyboard_axis','forward'), yaw_speed=getattr(args,'yaw_speed',.3), turn_forward_speed=getattr(args,'turn_forward_speed',0.), keyboard_motion_seconds=getattr(args,'keyboard_motion_seconds',None), continuous=getattr(args,'continuous',False), walk_speed=getattr(args,'walk_speed',.3), action_step_mode=getattr(args,'action_step_mode',None), left_speed=getattr(args,'left_speed',.3), speed_profile=getattr(args,'speed_profile','legacy'))
            attach_velocity_feedback(checked_trial, args, hashlib.sha256(Path(args.config).read_bytes()).hexdigest())
        elif args.stand_check:
            StandCheck(config, args.start_standing)
        else:
            ProneTest(config, args.test_joint, args.test_amplitude, args.duration, args.motor_test, args.response_test, args.joint_check)
    if args.check:
        print('真机验收配置检查通过；未初始化DDS。')
    elif args.analyze:
        with open(args.analyze, encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        joint = next((int(row['feedback']['test_joint_sdk']) for row in rows
                      if 'test_joint_sdk' in row.get('feedback', {})),
                     int(config.policy_to_sdk[config.data['policy']['joint_names'].index(args.test_joint)]))
        is_stand = any(row.get('mode') == 'fix_stand' for row in rows)
        print(json.dumps(assess_keyboard(rows,config) if any(r.get('feedback',{}).get('stand_profile')in ('keyboard23','keyboard29','keyboard35','keyboard_continuous') for r in rows) else assess_policy(rows, config) if any(r.get('feedback', {}).get('stand_profile') in ('policy16', 'forward19', 'keyboard23') for r in rows) else assess_stand(rows, config) if is_stand else assess_response(rows, joint), ensure_ascii=False, indent=2))
    elif args.restore_normal:
        if os.path.lexists(default_socket(0)):
            raise RuntimeError('验收停止接口仍存在，拒绝恢复原厂模式')
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(0, args.network)
        reader = VerifiedTransport(config, publish=False)
        deadline = time.monotonic() + 5
        while reader.snapshot() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError('恢复前无法读取LowState')
            time.sleep(0.01)
        print("等待连续一秒静止低姿态；此时不切换模式、不发送LowCmd。",flush=True)
        wait_resting_for_restore(config, reader)
        client = mode_client()
        if read_mode(client) not in ('', 'normal'):
            raise RuntimeError('未知运动模式，拒绝恢复')
        print('即将恢复normal，机器人可能自动站起；请准备App阻尼和卧倒。', flush=True)
        code, _ = client.SelectMode('normal')
        if code != 0:
            raise RuntimeError(f'恢复原厂模式请求失败: {code}')
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if read_mode(client) == 'normal':
                print('已确认原厂normal模式；请现场确认姿态，必要时用App卧倒。本程序未执行卧倒。')
                return
            time.sleep(0.1)
        raise RuntimeError('未确认原厂模式恢复；请检查App，不能认定恢复成功')
    elif args.serve:
        serve(args, config)
    else:
        run(args, config)


if __name__ == '__main__':
    main()
