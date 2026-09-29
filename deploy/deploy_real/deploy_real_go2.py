"""Go2真机部署程序：配置、状态机、安全、控制、DDS和记录。"""

import argparse
import json
import os
import select
import signal
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import yaml

from legged_gym import LEGGED_GYM_ROOT_DIR
from deploy.common.go2_policy import action_to_target, build_observation, projected_gravity, resolve_path


GO2_SERVO_MODE = 0x01  # Unitree Go2 low-level examples use PMSM/FOC mode 0x01.


class KeyboardVelocity:
    """终端按键产生短时速度目标；停止收到按键后自动归零。"""

    _KEYS = {"w": (0, 1), "s": (0, -1), "a": (1, 1), "d": (1, -1),
             "q": (2, 1), "e": (2, -1)}

    def __init__(self, limits, timeout=0.25):
        self.limits = _array(limits, "command_limits", size=3)
        if np.any(self.limits <= 0) or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("键盘速度上限与超时时间必须为正有限数")
        self.timeout = float(timeout)
        self.command = np.zeros(3, dtype=np.float32)
        self.expires_at = 0.0

    def feed(self, key, now):
        if key in ("x", " "):
            self.command.fill(0.0)
            self.expires_at = 0.0
        elif key in self._KEYS:
            axis, direction = self._KEYS[key]
            self.command.fill(0.0)
            self.command[axis] = direction * self.limits[axis]
            self.expires_at = float(now) + self.timeout

    def sample(self, now):
        if now >= self.expires_at:
            self.command.fill(0.0)
        return self.command.copy()


class TerminalKeys:
    """只从当前前台终端读取按键，并在退出时恢复终端设置。"""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.saved = None

    def start(self):
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def poll(self):
        pressed = []
        while select.select([self.fd], [], [], 0)[0]:
            data = os.read(self.fd, 32)
            if not data:
                break
            pressed.extend(data.decode("ascii", errors="ignore").lower())
        return pressed

    def close(self):
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            self.saved = None


def _array(values, name, size=12):
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name}必须是{size}个有限数值")
    return result


@dataclass(frozen=True)
class Go2RealConfig:
    """加载并校验Real专用go2.yaml。"""

    data: dict
    policy_path: str
    control_dt: float
    policy_to_sdk: np.ndarray
    sdk_to_policy: np.ndarray
    default_joint_pos: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    torque_limits: np.ndarray

    @classmethod
    def load(cls, path, root_dir):
        with open(path, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict) or data.get("robot") != "go2":
            raise ValueError("真机配置必须声明robot: go2")
        if data.get("format_version") != 1:
            raise ValueError("只支持format_version: 1的Go2真机配置")
        if (int(data["policy"]["num_actions"]), int(data["policy"]["num_observations"])) != (12, 45):
            raise ValueError("Go2真机部署只接受12动作/45观测策略")
        mapping = np.asarray(data["policy"]["policy_to_sdk"], dtype=np.int64)
        if sorted(mapping.tolist()) != list(range(12)):
            raise ValueError("policy_to_sdk必须是0到11的完整排列")
        control_dt = float(data["step_dt"])
        if not np.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("step_dt必须为正有限数")
        return cls(
            data=data,
            policy_path=resolve_path(data["policy_path"], root_dir),
            control_dt=control_dt,
            policy_to_sdk=mapping,
            sdk_to_policy=np.argsort(mapping),
            default_joint_pos=_array(data["actions"]["default_joint_pos"], "default_joint_pos"),
            stiffness=_array(data["control"]["stiffness"], "stiffness"),
            damping=_array(data["control"]["damping"], "damping"),
            torque_limits=_array(data["control"]["torque_limits"], "torque_limits"),
        )

    @property
    def safety(self):
        return self.data["safety"]

    def clip_command(self, command):
        command = _array(command, "command", size=3)
        limits = _array(self.data["command_limits"], "command_limits", size=3)
        return np.clip(command, -limits, limits)


class Go2Mode(str, Enum):
    PASSIVE = "passive"
    FIX_STAND = "fix_stand"
    POLICY = "policy"
    DAMPING = "damping"


class Go2FSM:
    """强制按Passive→FixStand→Policy顺序进入策略控制。"""

    def __init__(self, default_joint_pos, stand_duration):
        self.default_joint_pos = _array(default_joint_pos, "default_joint_pos")
        self.stand_duration = float(stand_duration)
        if self.stand_duration <= 0:
            raise ValueError("stand_duration必须为正数")
        self.mode = Go2Mode.PASSIVE
        self.mode_started_at = 0.0
        self.stand_start = None
        self.stand_complete = False
        self.fault_reason = ""

    def enter_fix_stand(self, now, joint_pos):
        if self.mode is not Go2Mode.PASSIVE:
            return False
        self.stand_start = _array(joint_pos, "joint_pos").copy()
        self.mode = Go2Mode.FIX_STAND
        self.mode_started_at = float(now)
        self.stand_complete = False
        return True

    def stand_target(self, now):
        if self.mode is not Go2Mode.FIX_STAND or self.stand_start is None:
            raise RuntimeError("当前不在FixStand状态")
        ratio = np.clip((float(now) - self.mode_started_at) / self.stand_duration, 0.0, 1.0)
        alpha = ratio * ratio * (3.0 - 2.0 * ratio)
        self.stand_complete = bool(ratio >= 1.0)
        return self.stand_start * (1.0 - alpha) + self.default_joint_pos * alpha

    def enter_policy(self, now):
        if self.mode is not Go2Mode.FIX_STAND or not self.stand_complete:
            return False
        self.mode = Go2Mode.POLICY
        self.mode_started_at = float(now)
        return True

    def enter_passive(self, now):
        if self.mode not in (Go2Mode.FIX_STAND, Go2Mode.DAMPING):
            return False
        self.mode = Go2Mode.PASSIVE
        self.mode_started_at = float(now)
        self.stand_start = None
        self.stand_complete = False
        self.fault_reason = ""
        return True

    def emergency_damping(self, now, reason):
        self.mode = Go2Mode.DAMPING
        self.mode_started_at = float(now)
        self.fault_reason = str(reason)


@dataclass(frozen=True)
class SafetyResult:
    safe: bool
    reason: str = ""


class Go2Safety:
    """检查状态和待发送策略命令，不依赖DDS。"""

    def __init__(self, config, stiffness, damping, torque_limits):
        self.state_timeout = float(config["state_timeout"])
        self.max_loop_dt = float(config["max_loop_dt"])
        self.max_roll = float(config["max_roll"])
        self.max_pitch = float(config["max_pitch"])
        self.max_joint_velocity = float(config["max_joint_velocity"])
        self.max_action = float(config["max_action"])
        self.max_action_delta = float(config["max_action_delta"])
        self.torque_limit_ratio = float(config["torque_limit_ratio"])
        self.state_joint_margin = float(config.get("state_joint_margin", 0.0))
        self.joint_lower = _array(config["joint_lower"], "joint_lower")
        self.joint_upper = _array(config["joint_upper"], "joint_upper")
        self.stiffness = _array(stiffness, "stiffness")
        self.damping = _array(damping, "damping")
        self.torque_limits = _array(torque_limits, "torque_limits")
        if np.any(self.joint_lower >= self.joint_upper):
            raise ValueError("关节下限必须小于上限")
        if not 0 < self.torque_limit_ratio <= 1:
            raise ValueError("torque_limit_ratio必须位于(0, 1]")

    @staticmethod
    def _fail(reason):
        return SafetyResult(False, reason)

    def check_state(self, now, state_time, roll, pitch, joint_pos, joint_vel, loop_dt=None):
        values = np.asarray([now, state_time, roll, pitch], dtype=np.float64)
        q = _array(joint_pos, "joint_pos")
        dq = _array(joint_vel, "joint_vel")
        if not np.isfinite(values).all():
            return self._fail("状态包含NaN或Inf")
        if now - state_time > self.state_timeout or now < state_time:
            return self._fail("LowState通信超时")
        if loop_dt is not None and (not np.isfinite(loop_dt) or loop_dt > self.max_loop_dt or loop_dt < 0):
            return self._fail("控制循环超时")
        if abs(roll) > self.max_roll or abs(pitch) > self.max_pitch:
            return self._fail("机身姿态超过安全限制")
        if np.any(q < self.joint_lower - self.state_joint_margin) or np.any(q > self.joint_upper + self.state_joint_margin):
            return self._fail("关节位置超过安全限制")
        if np.any(np.abs(dq) > self.max_joint_velocity):
            return self._fail("关节速度超过安全限制")
        return SafetyResult(True)

    def check_policy_command(self, action, previous_action, target, joint_pos, joint_vel, stiffness=None, damping=None):
        action = _array(action, "action")
        previous_action = _array(previous_action, "previous_action")
        target = _array(target, "target")
        q = _array(joint_pos, "joint_pos")
        dq = _array(joint_vel, "joint_vel")
        if np.any(np.abs(action) > self.max_action):
            return self._fail("策略动作超过安全限制")
        if np.any(np.abs(action - previous_action) > self.max_action_delta):
            return self._fail("策略动作单步变化过大")
        if np.any(target < self.joint_lower) or np.any(target > self.joint_upper):
            return self._fail("目标关节角超过安全限制")
        kp = self.stiffness if stiffness is None else _array(stiffness, "stiffness")
        kd = self.damping if damping is None else _array(damping, "damping")
        estimated_torque = kp * (target - q) - kd * dq
        if np.any(np.abs(estimated_torque) > self.torque_limits * self.torque_limit_ratio):
            return self._fail("估算PD力矩超过安全限制")
        return SafetyResult(True)

    def limit_target(self, target):
        return np.clip(_array(target, "target"), self.joint_lower, self.joint_upper)


@dataclass
class Go2State:
    timestamp: float
    quaternion_wxyz: np.ndarray
    angular_velocity: np.ndarray
    joint_pos_sdk: np.ndarray
    joint_vel_sdk: np.ndarray


@dataclass
class Go2Command:
    mode: Go2Mode
    target_pos_sdk: np.ndarray
    kp_sdk: np.ndarray
    kd_sdk: np.ndarray
    reason: str = ""


def quaternion_to_roll_pitch(quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
        raise ValueError("IMU四元数无效")
    w, x, y, z = q / np.linalg.norm(q)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)


class Go2Controller:
    """生成电机目标；不创建DDS发布器。"""

    def __init__(self, config, policy=None):
        self.config = config
        self.policy = policy if policy is not None else torch.jit.load(config.policy_path, map_location="cpu")
        if hasattr(self.policy, "eval"):
            self.policy.eval()
        fsm_config = config.data["fsm"]
        self.fsm = Go2FSM(config.default_joint_pos, fsm_config["stand_duration"])
        self.safety = Go2Safety(config.safety, config.stiffness, config.damping, config.torque_limits)
        self.stand_kp = _array(fsm_config["stand_kp"], "stand_kp")
        self.stand_kd = _array(fsm_config["stand_kd"], "stand_kd")
        self.last_action = np.zeros(12, dtype=np.float32)
        self.last_step_time = None
        self.command = np.zeros(3, dtype=np.float32)

    def validate_policy(self):
        with torch.inference_mode():
            output = self.policy(torch.zeros((1, 45), dtype=torch.float32))
        action = output.detach().cpu().numpy().reshape(-1)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError(f"策略输出无效: shape={action.shape}")
        return action.astype(np.float32)

    def set_velocity_command(self, command):
        self.command = self.config.clip_command(command)

    def _policy_joint_state(self, state):
        q_sdk = _array(state.joint_pos_sdk, "joint_pos_sdk")
        dq_sdk = _array(state.joint_vel_sdk, "joint_vel_sdk")
        return q_sdk[self.config.policy_to_sdk], dq_sdk[self.config.policy_to_sdk]

    def _to_sdk(self, policy_values):
        result = np.empty(12, dtype=np.float32)
        result[self.config.policy_to_sdk] = _array(policy_values, "policy_values")
        return result

    def request_fix_stand(self, now, state):
        q, _ = self._policy_joint_state(state)
        return self.fsm.enter_fix_stand(now, q)

    def request_policy(self, now):
        return self.fsm.enter_policy(now)

    def emergency_stop(self, now, reason="用户急停"):
        self.fsm.emergency_damping(now, reason)

    def _damping_command(self, state, reason=""):
        return Go2Command(
            mode=self.fsm.mode,
            target_pos_sdk=_array(state.joint_pos_sdk, "joint_pos_sdk").copy(),
            kp_sdk=np.zeros(12, dtype=np.float32),
            kd_sdk=np.full(12, float(self.config.data["fsm"]["damping_kd"]), dtype=np.float32),
            reason=reason,
        )

    def _policy_action(self, state, q, dq):
        observation = build_observation(
            state.angular_velocity,
            projected_gravity(state.quaternion_wxyz),
            self.command,
            q,
            dq,
            self.last_action,
            self.config.data,
        )
        with torch.inference_mode():
            tensor = torch.tensor(observation.tolist(), dtype=torch.float32).unsqueeze(0)
            output = self.policy(tensor)
        action = output.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return action, action_to_target(action, self.config.data)

    def step(self, now, state):
        try:
            q, dq = self._policy_joint_state(state)
            roll, pitch = quaternion_to_roll_pitch(state.quaternion_wxyz)
            loop_dt = None if self.last_step_time is None else float(now) - self.last_step_time
            result = self.safety.check_state(now, state.timestamp, roll, pitch, q, dq, loop_dt)
        except (ValueError, TypeError, IndexError) as error:
            self.fsm.emergency_damping(now, f"状态解析失败: {error}")
            return self._damping_command(state, self.fsm.fault_reason)
        finally:
            self.last_step_time = float(now)
        if not result.safe:
            self.fsm.emergency_damping(now, result.reason)
            return self._damping_command(state, result.reason)
        if self.fsm.mode in (Go2Mode.PASSIVE, Go2Mode.DAMPING):
            return self._damping_command(state, self.fsm.fault_reason)
        if self.fsm.mode is Go2Mode.FIX_STAND:
            target = self.safety.limit_target(self.fsm.stand_target(now))
            result = self.safety.check_policy_command(
                np.zeros(12), np.zeros(12), target, q, dq, self.stand_kp, self.stand_kd
            )
        else:
            try:
                action, target = self._policy_action(state, q, dq)
                result = self.safety.check_policy_command(action, self.last_action, target, q, dq)
            except (ValueError, RuntimeError, TypeError) as error:
                result = SafetyResult(False, f"策略推理失败: {error}")
            if result.safe:
                self.last_action = action
        if not result.safe:
            self.fsm.emergency_damping(now, result.reason)
            return self._damping_command(state, result.reason)
        target = self.safety.limit_target(target)
        kp = self.stand_kp if self.fsm.mode is Go2Mode.FIX_STAND else self.config.stiffness
        kd = self.stand_kd if self.fsm.mode is Go2Mode.FIX_STAND else self.config.damping
        return Go2Command(self.fsm.mode, self._to_sdk(target), self._to_sdk(kp), self._to_sdk(kd))


class Go2DDSTransport:
    """在LowState/LowCmd与Go2Controller之间转换。"""

    def __init__(self, config, publish=True):
        try:
            from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as error:
            raise RuntimeError("未安装unitree_sdk2py，请安装官方Python SDK") from error
        self._crc = CRC()
        self._lock = threading.Lock()
        self._state = None
        self._publish = publish
        self._low_cmd = unitree_go_msg_dds__LowCmd_()
        self._init_low_cmd()
        dds = config.data["dds"]
        self._subscriber = ChannelSubscriber(dds["lowstate_topic"], LowStateGo)
        self._subscriber.Init(self._on_state, int(dds["subscriber_queue"]))
        self._publisher = None
        if publish:
            self._foreign_lowcmd = threading.Event()
            self._lowcmd_probe = ChannelSubscriber(dds["lowcmd_topic"], LowCmdGo)
            self._lowcmd_probe.Init(lambda _message: self._foreign_lowcmd.set(), 1)
            time.sleep(0.25)
            if self._foreign_lowcmd.is_set():
                raise RuntimeError("检测到其他LowCmd发布者，请先关闭其他控制程序")
            self._publisher = ChannelPublisher(dds["lowcmd_topic"], LowCmdGo)
            self._publisher.Init()

    def _init_low_cmd(self):
        self._low_cmd.head[0] = 0xFE
        self._low_cmd.head[1] = 0xEF
        self._low_cmd.level_flag = 0xFF
        self._low_cmd.gpio = 0
        for motor in self._low_cmd.motor_cmd:
            motor.mode = GO2_SERVO_MODE
            motor.q = 2.146e9
            motor.dq = 16000.0
            motor.kp = motor.kd = motor.tau = 0.0

    def _on_state(self, message):
        state = Go2State(
            time.monotonic(),
            np.asarray(message.imu_state.quaternion, dtype=np.float32),
            np.asarray(message.imu_state.gyroscope, dtype=np.float32),
            np.asarray([message.motor_state[i].q for i in range(12)], dtype=np.float32),
            np.asarray([message.motor_state[i].dq for i in range(12)], dtype=np.float32),
        )
        with self._lock:
            self._state = state

    def snapshot(self):
        with self._lock:
            if self._state is None:
                return None
            state = Go2State(
                self._state.timestamp,
                self._state.quaternion_wxyz.copy(),
                self._state.angular_velocity.copy(),
                self._state.joint_pos_sdk.copy(),
                self._state.joint_vel_sdk.copy(),
            )
            return state

    def send(self, command):
        if not self._publish:
            return
        for i in range(12):
            motor = self._low_cmd.motor_cmd[i]
            motor.mode = GO2_SERVO_MODE
            motor.q = float(command.target_pos_sdk[i])
            motor.dq = 0.0
            motor.kp = float(command.kp_sdk[i])
            motor.kd = float(command.kd_sdk[i])
            motor.tau = 0.0
        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._publisher.Write(self._low_cmd)


class Go2Recorder:
    """记录Go2真机周期数据到JSONL。"""

    def __init__(self, config, root_dir):
        self.enabled = bool(config.get("enabled", True))
        self.flush_interval = int(config.get("flush_interval", 50))
        self.count = 0
        self.stream = None
        self.path = None
        if self.enabled:
            directory = resolve_path(config["directory"], root_dir)
            os.makedirs(directory, exist_ok=True)
            self.path = os.path.join(directory, time.strftime("go2_real_%Y%m%d_%H%M%S.jsonl"))
            self.stream = open(self.path, "x", encoding="utf-8")

    def write(self, now, state, command, velocity_command):
        if not self.enabled:
            return
        to_list = lambda value: np.asarray(value, dtype=np.float32).tolist()
        row = {
            "time": float(now), "state_time": float(state.timestamp),
            "mode": command.mode.value, "reason": command.reason,
            "quaternion_wxyz": to_list(state.quaternion_wxyz),
            "angular_velocity": to_list(state.angular_velocity),
            "joint_pos_sdk": to_list(state.joint_pos_sdk),
            "joint_vel_sdk": to_list(state.joint_vel_sdk),
            "velocity_command": to_list(velocity_command),
            "target_pos_sdk": to_list(command.target_pos_sdk),
            "kp_sdk": to_list(command.kp_sdk), "kd_sdk": to_list(command.kd_sdk),
        }
        self.stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.count += 1
        if self.count % self.flush_interval == 0:
            self.stream.flush()

    def close(self):
        if self.stream is not None:
            self.stream.flush()
            self.stream.close()
            self.stream = None


def run(args):
    if not args.check and args.network != "lo" and not args.read_only:
        raise RuntimeError("真机主动电机控制尚未验收，仅允许--read-only")
    if not args.check and not args.read_only and not args.simulation_auto:
        raise RuntimeError("主动控制仅支持lo回环网卡上的--simulation-auto")
    if args.keyboard and not sys.stdin.isatty():
        raise RuntimeError("--keyboard需要前台交互式终端")
    config = Go2RealConfig.load(os.path.abspath(args.config), LEGGED_GYM_ROOT_DIR)
    controller = Go2Controller(config)
    controller.validate_policy()
    if args.check:
        print("Real配置和TorchScript策略加载成功；未初始化DDS。")
        return 0
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    domain_id = int(config.data["dds"]["domain_id"] if args.domain_id is None else args.domain_id)
    if args.simulation_auto and (args.network != "lo" or domain_id == 0):
        raise ValueError("--simulation-auto仅允许在lo回环网卡且非0 DDS domain中使用")
    if args.shadow_policy and args.duration <= float(config.data["fsm"]["stand_duration"]):
        raise ValueError("--shadow-policy的--duration必须大于站姿插值时间")
    ChannelFactoryInitialize(domain_id, args.network)
    transport = Go2DDSTransport(config, publish=not args.read_only)
    recorder = Go2Recorder(config.data["recording"], LEGGED_GYM_ROOT_DIR)
    keyboard = TerminalKeys() if args.keyboard else None
    keyboard_velocity = KeyboardVelocity(config.data["command_limits"]) if args.keyboard else None
    stop = False

    def request_stop(_signum=None, _frame=None):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    deadline = time.monotonic() + args.connect_timeout
    while True:
        state = transport.snapshot()
        if state is not None:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("等待LowState超时；请检查网卡、DDS domain和机器人/仿真器")
        time.sleep(0.01)
    print("已连接LowState。")
    if args.read_only:
        print("当前为只读模式，不创建LowCmd发布器。")
        if args.shadow_policy:
            print("自动计算策略目标；可使用键盘输入。" if args.keyboard else "自动计算零速度策略目标。")
    else:
        print("DDS仿真自动站立并进入策略；Ctrl+C退出。")
    next_step = run_started_at = time.monotonic()
    auto_policy_attempted = False
    last_reason = None
    if args.simulation_auto or args.shadow_policy:
        controller.set_velocity_command([0.0, 0.0, 0.0] if args.shadow_policy else args.command)
        controller.request_fix_stand(next_step, state)
        print("只读策略影子检查：开始虚拟FixStand。" if args.shadow_policy else "DDS仿真自动验收：开始FixStand。")
    try:
        if keyboard:
            keyboard.start()
            print("键盘：W/S前后，A/D左右，Q/E转向，X或空格停止行走，Z退出并进入阻尼；输入超时0.25秒归零。")
        while not stop:
            input_now = time.monotonic()
            if keyboard:
                for key in keyboard.poll():
                    if key == "z":
                        controller.emergency_stop(input_now, "键盘请求阻尼退出")
                        stop = True
                    else:
                        keyboard_velocity.feed(key, input_now)
                if stop:
                    break
            state = transport.snapshot()
            if state is None:
                time.sleep(0.001)
                continue
            # DDS回调可能在复制状态时更新它；现在取时间，避免误判“未来状态”。
            now = time.monotonic()
            if args.simulation_auto or args.shadow_policy:
                controller.set_velocity_command(
                    keyboard_velocity.sample(now) if keyboard else ([0.0, 0.0, 0.0] if args.shadow_policy else args.command)
                )
                if not auto_policy_attempted and now - run_started_at >= float(config.data["fsm"]["stand_duration"]):
                    controller.step(now, state)
                    entered = controller.request_policy(now)
                    auto_policy_attempted = True
                    label = "只读策略影子检查" if args.shadow_policy else "DDS仿真自动验收"
                    print(f"{label}：进入Policy={entered}。")
                if now - run_started_at >= args.duration:
                    stop = True
            else:
                controller.set_velocity_command([0.0, 0.0, 0.0])
            command = controller.step(now, state)
            transport.send(command)
            recorder.write(now, state, command, controller.command)
            if command.reason and command.reason != last_reason:
                print(f"安全降级至{command.mode.value}: {command.reason}")
            last_reason = command.reason
            next_step += config.control_dt
            time.sleep(max(0.0, next_step - time.monotonic()))
    finally:
        if keyboard:
            keyboard.close()
        if not args.read_only:
            controller.emergency_stop(time.monotonic(), "程序退出")
            count = max(1, int(config.data["fsm"]["damping_duration"] / config.control_dt))
            for _ in range(count):
                latest_state = transport.snapshot()
                if latest_state is not None:
                    transport.send(controller.step(time.monotonic(), latest_state))
                time.sleep(config.control_dt)
        recorder.close()
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Go2安全真机部署入口")
    parser.add_argument("network", nargs="?", help="DDS网卡；仿真为lo，真机为有线网卡")
    parser.add_argument("--config", default=os.path.join(LEGGED_GYM_ROOT_DIR, "deploy/deploy_real/configs/go2.yaml"))
    parser.add_argument("--domain-id", type=int, help="覆盖配置中的DDS domain")
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--check", action="store_true", help="只检查Real配置与模型，不初始化DDS")
    parser.add_argument("--read-only", action="store_true", help="只接收LowState，不创建LowCmd发布器")
    parser.add_argument("--shadow-policy", action="store_true", help="仅配合--read-only，默认计算零速度策略目标")
    parser.add_argument("--simulation-auto", action="store_true", help="仅用于lo与非0 domain的DDS仿真")
    parser.add_argument("--keyboard", action="store_true", help="仅用于只读影子检查或DDS仿真；按键超时归零")
    parser.add_argument("--duration", type=float, default=10.0, help="自动仿真或只读影子检查的总时长")
    parser.add_argument("--command", type=float, nargs=3, default=[0.3, 0.0, 0.0], metavar=("VX", "VY", "YAW"))
    args = parser.parse_args()
    if not args.check and not args.network:
        parser.error("非--check模式必须提供network")
    if args.shadow_policy and (not args.read_only or args.simulation_auto or args.check):
        parser.error("--shadow-policy必须配合--read-only，且不能与--check或--simulation-auto同时使用")
    if args.keyboard and not (args.shadow_policy or args.simulation_auto):
        parser.error("--keyboard仅允许配合--shadow-policy或--simulation-auto")
    if not args.check and args.network != "lo" and not args.read_only:
        parser.error("真机主动电机控制尚未验收，仅允许--read-only")
    if not args.check and not args.read_only and not args.simulation_auto:
        parser.error("主动控制仅支持lo回环网卡上的--simulation-auto")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
