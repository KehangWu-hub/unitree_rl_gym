import argparse
import json
import time
from contextlib import nullcontext

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

from legged_gym import LEGGED_GYM_ROOT_DIR
from deploy.common.go2_policy import (
    action_to_target,
    build_observation,
    load_deploy_config,
    projected_gravity,
    resolve_path,
)


def load_runtime_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def joint_addresses(model, joint_names):
    qpos = np.asarray([int(model.joint(name).qposadr[0]) for name in joint_names], dtype=np.int64)
    qvel = np.asarray([int(model.joint(name).dofadr[0]) for name in joint_names], dtype=np.int64)
    actuators = []
    for name in joint_names:
        joint_id = int(model.joint(name).id)
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if matches.size != 1:
            raise ValueError(f"Expected one actuator for {name}, found {matches.size}")
        actuators.append(int(matches[0]))
    return qpos, qvel, np.asarray(actuators, dtype=np.int64)


def run(args):
    runtime = load_runtime_config(args.config)
    policy_path = resolve_path(args.policy or runtime["policy_path"], LEGGED_GYM_ROOT_DIR)
    deploy_path = resolve_path(args.deploy_config or runtime["deploy_config_path"], LEGGED_GYM_ROOT_DIR)
    xml_path = resolve_path(runtime["xml_path"], LEGGED_GYM_ROOT_DIR)
    deploy_cfg = load_deploy_config(deploy_path)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = float(args.simulation_dt or runtime["simulation_dt"])
    overrides = runtime.get("model_overrides", {})
    if "joint_damping" in overrides:
        model.dof_damping[6:] = float(overrides["joint_damping"])
    if "armature" in overrides:
        model.dof_armature[6:] = float(overrides["armature"])
    if "friction_loss" in overrides:
        model.dof_frictionloss[6:] = float(overrides["friction_loss"])
    if "sliding_friction" in overrides:
        model.geom_friction[:, 0] = float(overrides["sliding_friction"])
    step_dt = float(deploy_cfg["step_dt"])
    decimation = int(round(step_dt / model.opt.timestep))
    if decimation < 1 or not np.isclose(decimation * model.opt.timestep, step_dt):
        raise ValueError("Policy step_dt must be an integer multiple of simulation_dt")

    joint_names = deploy_cfg["policy"]["joint_names"]
    qpos_ids, qvel_ids, actuator_ids = joint_addresses(model, joint_names)
    default_pos = np.asarray(deploy_cfg["actions"]["default_joint_pos"], dtype=np.float32)
    kp = np.asarray(deploy_cfg["control"]["stiffness"], dtype=np.float32)
    kd = np.asarray(deploy_cfg["control"]["damping"], dtype=np.float32)
    torque_limits = np.asarray(deploy_cfg["control"]["torque_limits"], dtype=np.float32)
    actuator_limits = np.abs(model.actuator_ctrlrange[actuator_ids])[:, 1]
    torque_limits = np.minimum(torque_limits, actuator_limits)

    data.qpos[qpos_ids] = default_pos
    mujoco.mj_forward(model, data)

    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()
    command = np.asarray(args.command or runtime["command"], dtype=np.float32)
    command[0] = np.clip(command[0], *deploy_cfg["commands"]["lin_vel_x"])
    command[1] = np.clip(command[1], *deploy_cfg["commands"]["lin_vel_y"])
    command[2] = np.clip(command[2], *deploy_cfg["commands"]["ang_vel_yaw"])
    action = np.zeros(int(deploy_cfg["policy"]["num_actions"]), dtype=np.float32)
    target_pos = default_pos.copy()

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

    viewer_context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(model, data)
    start = time.perf_counter()
    with viewer_context as viewer:
        for counter in range(total_steps):
            step_start = time.perf_counter()
            sim_time = counter * model.opt.timestep
            if push_force is not None and args.push_at <= sim_time < args.push_at + args.push_duration:
                data.xfrc_applied[base_body_id, :3] = push_force
            else:
                data.xfrc_applied[base_body_id, :3] = 0.0
            joint_pos = data.qpos[qpos_ids].copy()
            joint_vel = data.qvel[qvel_ids].copy()
            torque = kp * (target_pos - joint_pos) - kd * joint_vel
            torque = np.clip(torque, -torque_limits, torque_limits)
            data.ctrl[actuator_ids] = torque
            max_abs_torque = max(max_abs_torque, float(np.max(np.abs(torque))))
            mujoco.mj_step(model, data)

            if counter % decimation == 0:
                quaternion = data.sensor("imu_quat").data.copy()
                gravity = projected_gravity(quaternion)
                ang_vel = data.sensor("imu_gyro").data.copy()
                obs = build_observation(
                    ang_vel, gravity, command, data.qpos[qpos_ids], data.qvel[qvel_ids], action, deploy_cfg
                )
                with torch.inference_mode():
                    action = policy(torch.from_numpy(obs).unsqueeze(0)).cpu().numpy().reshape(-1)
                target_pos = action_to_target(action, deploy_cfg)

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
            if realtime:
                remaining = model.opt.timestep - (time.perf_counter() - step_start)
                if remaining > 0:
                    time.sleep(remaining)

    elapsed = time.perf_counter() - start
    mean_body_velocity = np.mean(body_velocities, axis=0) if body_velocities else np.full(3, np.nan)
    velocity_error = mean_body_velocity[:2] - command[:2]
    yaw_rate_error = (float(np.mean(yaw_rates)) - float(command[2])) if yaw_rates else float("nan")
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
