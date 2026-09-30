"""DDS仿真的独立指令看门狗；停止锁存后不能恢复主动控制。"""

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time

from legged_gym import LEGGED_GYM_ROOT_DIR


COMMAND_TIMEOUT = 0.10


def default_socket(domain_id):
    return f"/tmp/go2-watchdog-{os.getuid()}-{domain_id}.sock"


def require_simulation(network, domain_id, duration):
    if network != "lo" or not 1 <= domain_id <= 232:
        raise ValueError("看门狗主动控制仅允许lo网卡和1至232的DDS domain")
    if not math.isfinite(duration) or not 0 < duration <= 60:
        raise ValueError("仿真主动控制时长必须在(0, 60]秒内")


def request(path, message, timeout=1.0):
    """请求本机看门狗，收到确认后才报告成功。"""
    with tempfile.TemporaryDirectory(prefix="go2-stop-") as directory:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
            client.bind(os.path.join(directory, "reply.sock"))
            client.settimeout(timeout)
            client.sendto(json.dumps(message).encode(), path)
            return json.loads(client.recv(4096))


class CommandGate:
    """只接受新鲜命令；独立检查状态、增益、目标角和估算力矩。"""

    def __init__(self, config, duration, motor_test=False):
        from deploy.deploy_real.deploy_real_go2 import Go2Safety
        self.config = config
        self.duration = duration
        self.motor_test = motor_test
        self.safety = Go2Safety(config.safety, config.stiffness, config.damping, config.torque_limits)
        self.command = None
        self.command_time = None
        self.started_at = None
        self.initial_q = None
        self.reason = ""
        self.stopped_at = None

    def stop(self, now, reason):
        if not self.reason:
            self.reason = str(reason)
            self.stopped_at = now

    def accept(self, payload, now, state):
        from deploy.deploy_real.deploy_real_go2 import Go2Command, Go2Mode, _array
        if self.reason:
            return
        try:
            timestamp = float(payload["time"])
            if not math.isfinite(timestamp) or not 0 <= now - timestamp <= COMMAND_TIMEOUT:
                raise ValueError("控制命令过期或时间无效")
            if self.command_time is not None and timestamp <= self.command_time:
                raise ValueError("控制命令时间未递增")
            command = Go2Command(
                Go2Mode(payload["mode"]), _array(payload["target"], "target"),
                _array(payload["kp"], "kp"), _array(payload["kd"], "kd"),
                str(payload.get("reason", "")),
            )
            allowed = (Go2Mode.MOTOR_TEST,) if self.motor_test else (Go2Mode.FIX_STAND, Go2Mode.POLICY)
            if command.mode in (Go2Mode.PASSIVE, Go2Mode.DAMPING):
                self.stop(now, command.reason or "控制程序请求阻尼")
                return
            if command.mode not in allowed:
                raise ValueError("控制模式与启动模式不一致")
            if self.initial_q is None:
                initial_state = _array(state.joint_pos_sdk, "initial_q")
                if self.motor_test and max(abs(command.target_pos_sdk - initial_state)) > 0.02:
                    raise ValueError("电机测试初始目标与实测姿态相差超过0.02弧度")
                # 首帧目标就是控制器采样的起始姿态；避免两个进程采样时差被当成其他关节动作。
                self.initial_q = command.target_pos_sdk.copy()
            self.command = command
            self.command_time = timestamp
            if self.started_at is None:
                self.started_at = now
        except (KeyError, ValueError, TypeError, OverflowError) as error:
            self.stop(now, f"无效控制命令: {error}")

    def output(self, now, state):
        import numpy as np
        from deploy.deploy_real.deploy_real_go2 import Go2Command, Go2Mode, _array, quaternion_to_roll_pitch
        if self.started_at is not None and not self.reason:
            if now - self.started_at >= self.duration:
                self.stop(now, "独立运行时长上限")
            elif now - self.command_time >= COMMAND_TIMEOUT:
                self.stop(now, "控制命令心跳超时")
            else:
                try:
                    q = _array(state.joint_pos_sdk, "q")[self.config.policy_to_sdk]
                    dq = _array(state.joint_vel_sdk, "dq")[self.config.policy_to_sdk]
                    roll, pitch = quaternion_to_roll_pitch(state.quaternion_wxyz)
                    result = self.safety.check_state(now, state.timestamp, roll, pitch, q, dq)
                    if not result.safe:
                        raise ValueError(result.reason)
                    cmd = self.command
                    kp = cmd.kp_sdk[self.config.policy_to_sdk]
                    kd = cmd.kd_sdk[self.config.policy_to_sdk]
                    target = cmd.target_pos_sdk[self.config.policy_to_sdk]
                    kp_max = 5.0 if self.motor_test else np.maximum(
                        self.config.stiffness, _array(self.config.data["fsm"]["stand_kp"], "stand_kp")
                    )
                    kd_max = 0.5 if self.motor_test else np.maximum(
                        self.config.damping, _array(self.config.data["fsm"]["stand_kd"], "stand_kd")
                    )
                    if np.any(kp < 0) or np.any(kp > kp_max) or np.any(kd < 0) or np.any(kd > kd_max):
                        raise ValueError("控制增益超过限制")
                    result = self.safety.check_policy_command(
                        np.zeros(12), np.zeros(12), target, q, dq, kp, kd
                    )
                    if not result.safe:
                        raise ValueError(result.reason)
                    if self.motor_test:
                        if np.max(np.abs(cmd.target_pos_sdk - self.initial_q)) > 0.02001:
                            raise ValueError("电机测试目标位移超过0.02弧度")
                        if np.count_nonzero(np.abs(cmd.target_pos_sdk - self.initial_q) > 1e-6) > 1:
                            raise ValueError("电机测试只能改变一个关节目标")
                        if np.max(np.abs(state.joint_pos_sdk - self.initial_q)) > 0.04:
                            raise ValueError("电机测试实际位移超过0.04弧度")
                        if np.max(np.abs(dq)) > 2.0 or np.max(np.abs(kp * (target - q) - kd * dq)) > 1.0:
                            raise ValueError("电机测试速度或估算力矩超过限制")
                except (ValueError, TypeError, AttributeError) as error:
                    self.stop(now, f"看门狗安全检查: {error}")
        if self.reason:
            # Kp=0，位置目标不参与力矩；即使状态无效也能生成有限阻尼命令。
            return Go2Command(Go2Mode.DAMPING, np.zeros(12, dtype=np.float32),
                             np.zeros(12, dtype=np.float32),
                             np.full(12, float(self.config.data["fsm"]["damping_kd"]), dtype=np.float32),
                             self.reason)
        return self.command


class WatchdogTransport:
    """主程序只订阅状态；独立子进程是唯一的LowCmd发布者。"""

    def __init__(self, config, config_path, network, domain_id, duration, connect_timeout, motor_test=False):
        from deploy.deploy_real.deploy_real_go2 import Go2DDSTransport
        require_simulation(network, domain_id, duration)
        self.path = default_socket(domain_id)
        if os.path.lexists(self.path):
            raise RuntimeError(f"看门狗接口已存在: {self.path}；请先确认旧进程状态")
        self.reader = Go2DDSTransport(config, publish=False)
        self.client_directory = tempfile.TemporaryDirectory(prefix="go2-controller-")
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.bind(os.path.join(self.client_directory.name, "reply.sock"))
        self.socket.setblocking(False)
        command = [sys.executable, "-m", "deploy.deploy_real.go2_watchdog", "--serve",
                   "--config", config_path, "--domain-id", str(domain_id),
                   "--duration", str(duration), "--connect-timeout", str(connect_timeout)]
        if motor_test:
            command.append("--motor-test")
        self.process = subprocess.Popen(command, cwd=LEGGED_GYM_ROOT_DIR, start_new_session=True)
        deadline = time.monotonic() + connect_timeout + 10.0
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("独立看门狗启动失败，未开始主动控制")
                try:
                    if request(self.path, {"type": "status"}, timeout=0.1).get("ready"):
                        return
                except (OSError, ValueError):
                    time.sleep(0.02)
            raise TimeoutError("等待独立看门狗启动超时")
        except BaseException:
            self.close()
            raise

    def snapshot(self):
        return self.reader.snapshot()

    def send(self, command):
        while True:
            try:
                if self.socket.recv(32) == b"stopped":
                    return False
            except BlockingIOError:
                break
        status = self.process.poll()
        if status == 0:
            return False
        if status is not None:
            raise RuntimeError("独立看门狗已退出，主动控制终止")
        payload = {"type": "command", "time": time.monotonic(), "mode": command.mode.value,
                   "target": command.target_pos_sdk.tolist(), "kp": command.kp_sdk.tolist(),
                   "kd": command.kd_sdk.tolist(), "reason": command.reason}
        try:
            self.socket.sendto(json.dumps(payload, allow_nan=False).encode(), self.path)
        except FileNotFoundError:
            if self.process.wait(timeout=1.0) == 0:
                return False
            raise
        return True

    def close(self):
        try:
            if self.process.poll() is None:
                try:
                    request(self.path, {"type": "stop"}, timeout=0.5)
                except OSError:
                    # 没有确认时，停止发送命令；独立进程会由命令超时进入阻尼。
                    pass
                try:
                    self.process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    raise RuntimeError(f"看门狗未按时退出，请检查进程{self.process.pid}及{self.path}")
        finally:
            self.socket.close()
            self.client_directory.cleanup()


def serve(args):
    import numpy as np
    from deploy.deploy_real.deploy_real_go2 import Go2DDSTransport, Go2RealConfig, Go2Recorder
    require_simulation("lo", args.domain_id, args.duration)
    if args.motor_test and not 1 <= args.duration <= 3:
        raise ValueError("受限电机测试时长必须在1至3秒内")
    config = Go2RealConfig.load(os.path.abspath(args.config), LEGGED_GYM_ROOT_DIR)
    path = default_socket(args.domain_id)
    gate = CommandGate(config, args.duration, args.motor_test)
    recorder = None
    transport = None
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as server:
        # 不移除已有接口，防止第二个控制程序接管正在使用的停止接口。
        server.bind(path)
        os.chmod(path, 0o600)
        server.setblocking(False)
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            ChannelFactoryInitialize(args.domain_id, "lo")
            transport = Go2DDSTransport(config, publish=True)
            deadline = time.monotonic() + args.connect_timeout
            while transport.snapshot() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("看门狗等待LowState超时")
                time.sleep(0.01)
            recording = dict(config.data["recording"])
            recording["directory"] += "/watchdog"
            recorder = Go2Recorder(recording, LEGGED_GYM_ROOT_DIR)
            signal.signal(signal.SIGINT, lambda *_: gate.stop(time.monotonic(), "看门狗收到停止信号"))
            signal.signal(signal.SIGTERM, lambda *_: gate.stop(time.monotonic(), "看门狗收到停止信号"))
            idle_deadline = time.monotonic() + args.connect_timeout
            last_reason = ""
            next_step = time.monotonic()
            while True:
                state = transport.snapshot()
                acknowledgements = []
                command_replies = []
                for _ in range(32):
                    try:
                        raw, address = server.recvfrom(8192)
                    except BlockingIOError:
                        break
                    try:
                        payload = json.loads(raw)
                        if payload["type"] == "stop":
                            gate.stop(time.monotonic(), "独立终端请求停止")
                            acknowledgements.append(address)
                        elif payload["type"] == "status":
                            server.sendto(json.dumps({"ready": True, "stopped": bool(gate.reason),
                                                      "reason": gate.reason, "pid": os.getpid()}).encode(), address)
                        elif payload["type"] == "command":
                            gate.accept(payload, time.monotonic(), state)
                            if address:
                                command_replies.append(address)
                        else:
                            raise ValueError("未知请求类型")
                    except (ValueError, KeyError, TypeError):
                        gate.stop(time.monotonic(), "看门狗收到无效请求")
                    except OSError:
                        # 状态查询客户端提前退出不影响主动控制。
                        pass
                now = time.monotonic()
                command = gate.output(now, state)
                if command is not None:
                    transport.send(command)
                    recorder.write(now, state, command, np.zeros(3))
                for address in acknowledgements:
                    if address:
                        try:
                            server.sendto(json.dumps({"stopped": True, "reason": gate.reason}).encode(), address)
                        except OSError:
                            pass
                for address in command_replies:
                    try:
                        server.sendto(b"stopped" if gate.reason else b"running", address)
                    except OSError:
                        pass
                if gate.reason and gate.reason != last_reason:
                    print(f"独立看门狗进入阻尼: {gate.reason}", flush=True)
                    last_reason = gate.reason
                if gate.reason and now - gate.stopped_at >= float(config.data["fsm"]["damping_duration"]):
                    break
                if gate.started_at is None and not gate.reason and now >= idle_deadline:
                    break
                next_step += config.control_dt
                time.sleep(max(0.0, next_step - time.monotonic()))
        finally:
            try:
                # 独立进程自身发生可捕获异常时，也尝试发送完整的一秒阻尼。
                if transport is not None and gate.started_at is not None and (
                        gate.stopped_at is None or time.monotonic() - gate.stopped_at < float(config.data["fsm"]["damping_duration"])):
                    gate.stop(time.monotonic(), "看门狗异常退出")
                    until = time.monotonic() + float(config.data["fsm"]["damping_duration"])
                    while time.monotonic() < until:
                        command = gate.output(time.monotonic(), transport.snapshot())
                        transport.send(command)
                        time.sleep(config.control_dt)
            finally:
                if recorder:
                    recorder.close()
                os.unlink(path)


def main():
    parser = argparse.ArgumentParser(description="Go2 DDS仿真独立停止接口")
    parser.add_argument("--stop", action="store_true", help="请求正在运行的仿真看门狗进入阻尼")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--config", default=os.path.join(LEGGED_GYM_ROOT_DIR, "deploy/deploy_real/configs/go2.yaml"))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--motor-test", action="store_true")
    args = parser.parse_args()
    if args.stop == args.serve:
        parser.error("必须指定--stop")
    if args.stop:
        response = request(default_socket(args.domain_id), {"type": "stop"})
        if not response.get("stopped"):
            raise RuntimeError("没有收到停止确认")
        print("独立看门狗已锁存停止并发送阻尼指令。")
    else:
        serve(args)


if __name__ == "__main__":
    main()
