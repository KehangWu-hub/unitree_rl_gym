"""仅供测试：本地Go2 MJCF通过官方SDK消息与被测部署程序交换状态。"""

import argparse
import json
import os
import signal
import threading
import time

import mujoco
import numpy as np

from legged_gym import LEGGED_GYM_ROOT_DIR
from deploy.deploy_mujoco.deploy_go2 import joint_addresses
from deploy.deploy_real.deploy_real_go2 import Go2RealConfig
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-id", type=int, required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--supported", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.domain_id <= 232:
        raise ValueError("fixture只允许非0仿真domain")
    cfg = Go2RealConfig.load(os.path.join(LEGGED_GYM_ROOT_DIR, "deploy/deploy_real/configs/go2.yaml"), LEGGED_GYM_ROOT_DIR)
    model = mujoco.MjModel.from_xml_path(os.path.join(LEGGED_GYM_ROOT_DIR, "resources/robots/go2/mjcf/scene_flat.xml"))
    model.opt.timestep = 0.002
    model.dof_damping[6:] = 0
    model.dof_armature[6:] = 0
    model.dof_frictionloss[6:] = 0
    model.geom_friction[:, 0] = 1.0
    if args.supported:
        model.opt.gravity[:] = 0  # 隔离关节方向/映射；不能用来证明真机承重稳定。
    data = mujoco.MjData(model)
    qpos, qvel, actuator = joint_addresses(model, cfg.data["policy"]["sdk_joint_names"])
    initial = cfg.default_joint_pos[cfg.sdk_to_policy]
    data.qpos[qpos] = initial
    if args.supported:
        data.qpos[2] = 0.6
    mujoco.mj_forward(model, data)
    base_pose = data.qpos[:7].copy()
    command = None
    lock = threading.Lock()
    crc = CRC()
    trace = open(os.path.join(args.directory, "trace.jsonl"), "w", buffering=1)

    def on_command(message):
        nonlocal command
        received = time.monotonic()
        valid = message.crc == crc.Crc(message)
        row = {"type": "command", "time": received, "crc_valid": valid,
               "mode": [m.mode for m in message.motor_cmd[:12]],
               "target": [m.q for m in message.motor_cmd[:12]],
               "kp": [m.kp for m in message.motor_cmd[:12]],
               "kd": [m.kd for m in message.motor_cmd[:12]],
               "tau": [m.tau for m in message.motor_cmd[:12]]}
        with lock:
            trace.write(json.dumps(row) + "\n")
            if valid:
                command = row

    ChannelFactoryInitialize(args.domain_id, "lo")
    subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
    subscriber.Init(on_command, 10)
    publisher = ChannelPublisher("rt/lowstate", LowState_)
    publisher.Init()
    state = unitree_go_msg_dds__LowState_()
    stop = False

    def stop_signal(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    with open(os.path.join(args.directory, "ready"), "w") as stream:
        stream.write(str(os.getpid()))
    next_step = time.monotonic()
    counter = 0
    try:
        while not stop:
            with lock:
                latest = command
            if latest is None:
                torque = 20.0 * (initial - data.qpos[qpos]) - 0.5 * data.qvel[qvel]
            else:
                torque = np.asarray(latest["kp"]) * (np.asarray(latest["target"]) - data.qpos[qpos]) - np.asarray(latest["kd"]) * data.qvel[qvel]
            data.ctrl[actuator] = np.clip(torque, -23.7, 23.7)
            mujoco.mj_step(model, data)
            if args.supported:
                data.qpos[:7] = base_pose
                data.qvel[:6] = 0
                mujoco.mj_forward(model, data)
            state.imu_state.quaternion = data.sensor("imu_quat").data.tolist()
            state.imu_state.gyroscope = data.sensor("imu_gyro").data.tolist()
            for i in range(12):
                state.motor_state[i].q = float(data.qpos[qpos[i]])
                state.motor_state[i].dq = float(data.qvel[qvel[i]])
            if not os.path.exists(os.path.join(args.directory, "pause_state")):
                publisher.Write(state)
            if counter % 5 == 0:
                with lock:
                    trace.write(json.dumps({"type": "state", "time": time.monotonic(),
                                            "height": float(data.qpos[2]), "q": data.qpos[qpos].tolist(),
                                            "max_torque": float(np.max(np.abs(torque)))}) + "\n")
            counter += 1
            next_step += model.opt.timestep
            time.sleep(max(0, next_step - time.monotonic()))
    finally:
        subscriber.Close()
        publisher.Close()
        trace.close()


if __name__ == "__main__":
    main()
