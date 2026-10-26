import argparse
import json
import time
from contextlib import nullcontext

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

# 训练仓库的根目录，用来展开 YAML 路径中的 {LEGGED_GYM_ROOT_DIR}。
from legged_gym import LEGGED_GYM_ROOT_DIR
# MuJoCo 和真机共用的观测构造、姿态转换及动作解码函数。
from deploy.common.go2_policy import (
    PolicyObservationHistory,
    action_to_target,
    build_observation,
    load_deploy_config,
    projected_gravity,
    resolve_path,
)

def load_runtime_config(path):
    """读取仿真运行参数（configs/go2.yaml），不是训练导出的 deploy.yaml。"""
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def apply_model_overrides(model, overrides):
    """Apply explicit simulation parameters, including all friction components."""
    targets = {"joint_damping": model.dof_damping[6:],
               "armature": model.dof_armature[6:],
               "friction_loss": model.dof_frictionloss[6:],
               "sliding_friction": model.geom_friction[:, 0],
               "torsional_friction": model.geom_friction[:, 1],
               "rolling_friction": model.geom_friction[:, 2]}
    for name, value in overrides.items():
        if name not in targets or not np.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"Invalid MuJoCo model override: {name}={value}")
    for name, value in overrides.items():
        targets[name][:] = float(value)


def joint_addresses(model, joint_names):
    """按策略的关节顺序，找出每个关节在 MuJoCo 数据数组中的编号。"""
    # data.qpos 存当前关节角，data.qvel 存当前关节速度。
    # 这里得到的编号让后续代码能按策略顺序读取这 12 个关节，而不必猜 MuJoCo 的排列顺序。
    qpos = np.asarray([int(model.joint(name).qposadr[0]) for name in joint_names], dtype=np.int64)
    qvel = np.asarray([int(model.joint(name).dofadr[0]) for name in joint_names], dtype=np.int64)
    # data.ctrl 存要发给电机的控制量；逐个找出驱动这些关节的电机编号。
    actuators = []
    for name in joint_names:
        joint_id = int(model.joint(name).id)
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if matches.size != 1:
            raise ValueError(f"Expected one actuator for {name}, found {matches.size}")
        actuators.append(int(matches[0]))
    return qpos, qvel, np.asarray(actuators, dtype=np.int64)


def run(args):
    # 运行配置给出模型、策略和导出参数的路径；命令行参数可覆盖其中的策略路径。
    runtime = load_runtime_config(args.config)
    policy_path = resolve_path(args.policy or runtime["policy_path"], LEGGED_GYM_ROOT_DIR)
    deploy_path = resolve_path(args.deploy_config or runtime["deploy_config_path"], LEGGED_GYM_ROOT_DIR)
    xml_path = resolve_path(runtime["xml_path"], LEGGED_GYM_ROOT_DIR)
    deploy_cfg = load_deploy_config(deploy_path)

    # XML 定义机器人和场景，model 保存结构，data 保存随仿真变化的状态。
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = float(args.simulation_dt or runtime["simulation_dt"])
    # 覆盖 MJCF 的部分物理参数，使本次仿真尽量匹配训练环境。
    overrides = runtime.get("model_overrides", {})
    apply_model_overrides(model, overrides)
    # 物理仿真每 simulation_dt 秒走一步；策略每 step_dt 秒才计算一次动作。
    # 例如 0.002 秒与 0.02 秒对应：每 10 次物理步运行一次策略。
    step_dt = float(deploy_cfg["step_dt"])
    decimation = int(round(step_dt / model.opt.timestep))
    if decimation < 1 or not np.isclose(decimation * model.opt.timestep, step_dt):
        raise ValueError("Policy step_dt must be an integer multiple of simulation_dt")

    joint_names = deploy_cfg["policy"]["joint_names"]
    qpos_ids, qvel_ids, actuator_ids = joint_addresses(model, joint_names)
    # 控制增益和动作限幅来自训练导出的 deploy.yaml。
    default_pos = np.asarray(deploy_cfg["actions"]["default_joint_pos"], dtype=np.float32)
    kp = np.asarray(deploy_cfg["control"]["stiffness"], dtype=np.float32)
    kd = np.asarray(deploy_cfg["control"]["damping"], dtype=np.float32)
    torque_limits = np.asarray(deploy_cfg["control"]["torque_limits"], dtype=np.float32)
    actuator_limits = np.abs(model.actuator_ctrlrange[actuator_ids])[:, 1]
    torque_limits = np.minimum(torque_limits, actuator_limits)

    # 从默认关节姿态开始，更新由此推导的 MuJoCo 传感器和运动学状态。
    data.qpos[qpos_ids] = default_pos
    mujoco.mj_forward(model, data)

    # 加载 Isaac Gym 训练后导出的 TorchScript 策略，仅用于推理。
    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()
    # 速度指令依次为前进、侧向和偏航角速度，限制在训练时的指令范围内。
    command = np.asarray(args.command or runtime["command"], dtype=np.float32)
    command[0] = np.clip(command[0], *deploy_cfg["commands"]["lin_vel_x"])
    command[1] = np.clip(command[1], *deploy_cfg["commands"]["lin_vel_y"])
    command[2] = np.clip(command[2], *deploy_cfg["commands"]["ang_vel_yaw"])
    # 初始“上一次动作”为零；策略第一次运行前的关节目标为默认姿态。
    action = np.zeros(int(deploy_cfg["policy"]["num_actions"]), dtype=np.float32)
    target_pos = default_pos.copy()
    history = PolicyObservationHistory(deploy_cfg)

    duration = float(args.duration or runtime["simulation_duration"])
    total_steps = int(round(duration / model.opt.timestep))
    fall_height = float(runtime["fall_height"])
    fall_tilt = float(runtime["fall_tilt"])
    realtime = bool(runtime.get("realtime", True)) and not args.no_realtime
    heights, tilts, body_velocities, yaw_rates, max_abs_torque = [], [], [], [], 0.0
    fell = False
    initial_position = data.qpos[:3].copy()
    base_body_id = int(model.body("base_link").id)
    push_force = np.asarray(args.push_force, dtype=np.float64) if args.push_force else None

    # --headless 时不打开窗口；两种模式执行相同的物理和策略循环。
    viewer_context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(model, data)
    start = time.perf_counter()
    with viewer_context as viewer:
        for counter in range(total_steps):
            step_start = time.perf_counter()
            sim_time = counter * model.opt.timestep
            # 可选的外力测试：只在指定的仿真时间段给机身施力。
            if push_force is not None and args.push_at <= sim_time < args.push_at + args.push_duration:
                data.xfrc_applied[base_body_id, :3] = push_force
            else:
                data.xfrc_applied[base_body_id, :3] = 0.0
            # 每个物理步都使用当前关节状态执行 PD 控制，并将力矩送给 MuJoCo。
            joint_pos = data.qpos[qpos_ids].copy()
            joint_vel = data.qvel[qvel_ids].copy()
            torque = kp * (target_pos - joint_pos) - kd * joint_vel
            torque = np.clip(torque, -torque_limits, torque_limits)
            data.ctrl[actuator_ids] = torque
            max_abs_torque = max(max_abs_torque, float(np.max(np.abs(torque))))
            mujoco.mj_step(model, data)

            # 策略运行频率低于物理仿真频率；两次推理间沿用上次的 target_pos。
            if counter % decimation == 0:
                # 从 IMU 和关节读取状态；观测顺序和缩放由共用函数及 deploy.yaml 决定。
                quaternion = data.sensor("imu_quat").data.copy()
                gravity = projected_gravity(quaternion)
                ang_vel = data.sensor("imu_gyro").data.copy()
                obs = build_observation(
                    ang_vel, gravity, command, data.qpos[qpos_ids], data.qvel[qvel_ids], action, deploy_cfg
                )
                obs = history.append(obs)
                # HIM输入270维历史；旧策略输入45维单帧。
                with torch.inference_mode():
                    action = policy(torch.from_numpy(obs).unsqueeze(0)).cpu().numpy().reshape(-1)
                target_pos = action_to_target(action, deploy_cfg)
                if deploy_cfg.get("format_version") == 2:
                    action = np.clip(action, -deploy_cfg["actions"]["clip"], deploy_cfg["actions"]["clip"])

                # 记录稳定性指标；高度过低或倾斜过大时提前结束仿真。
                height = float(data.qpos[2])
                tilt = float(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))
                heights.append(height)
                tilts.append(tilt)
                rotation_world_from_body = data.xmat[base_body_id].reshape(3, 3)
                body_velocities.append(rotation_world_from_body.T @ data.qvel[:3])
                yaw_rates.append(float(ang_vel[2]))
                if height < fall_height or tilt > fall_tilt:
                    fell = True
                    break

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
            # 用墙上时钟补足每一步的时间，仅影响播放速度，不改变仿真步长。
            if realtime:
                remaining = model.opt.timestep - (time.perf_counter() - step_start)
                if remaining > 0:
                    time.sleep(remaining)

    elapsed = time.perf_counter() - start
    mean_body_velocity = np.mean(body_velocities, axis=0) if body_velocities else np.full(3, np.nan)
    velocity_error = mean_body_velocity[:2] - command[:2]
    yaw_rate_error = (float(np.mean(yaw_rates)) - float(command[2])) if yaw_rates else float("nan")
    # 输出位移、速度误差和是否跌倒等结果，便于比较不同策略或指令。
    metrics = {
        "fell": fell,
        "simulated_seconds": (counter + 1) * model.opt.timestep,
        "wall_seconds": elapsed,
        "final_height": heights[-1] if heights else None,
        "min_height": min(heights) if heights else None,
        "max_tilt_rad": max(tilts) if tilts else None,
        "max_abs_torque": max_abs_torque,
        "displacement_xy": (data.qpos[:2] - initial_position[:2]).tolist(),
        "mean_body_velocity": mean_body_velocity.tolist(),
        "velocity_error_xy": velocity_error.tolist(),
        "mean_yaw_rate": float(np.mean(yaw_rates)) if yaw_rates else None,
        "yaw_rate_error": yaw_rate_error,
        "finite": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(action).all()),
    }
    print(json.dumps(metrics, indent=2))
    return 2 if fell or not metrics["finite"] else 0


if __name__ == "__main__":
    # 从命令行接收运行参数；不指定时使用 configs/go2.yaml 中的默认值。
    parser = argparse.ArgumentParser(description="Run a Go2 TorchScript policy in MuJoCo")
    parser.add_argument("--config", default=f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/go2.yaml")
    parser.add_argument("--policy", help="Override policy_path from the runtime config")
    parser.add_argument("--deploy-config", help="Override deploy_config_path from the runtime config")
    parser.add_argument("--duration", type=float, help="Override simulation duration")
    parser.add_argument("--simulation-dt", type=float, help="Override the MuJoCo physics time step")
    parser.add_argument("--command", type=float, nargs=3, metavar=("VX", "VY", "WZ"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--push-at", type=float, default=5.0, help="Push start time in simulated seconds")
    parser.add_argument("--push-duration", type=float, default=0.2, help="Push duration in simulated seconds")
    parser.add_argument("--push-force", type=float, nargs=3, metavar=("FX", "FY", "FZ"))
    raise SystemExit(run(parser.parse_args()))
