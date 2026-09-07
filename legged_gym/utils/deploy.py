import os

import yaml


GO2_SDK_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


def _tensor_list(value):
    return value.detach().cpu().reshape(-1).tolist()


def export_deploy_config(env, path, robot_name):
    """Export the runtime contract from the instantiated training environment."""
    os.makedirs(path, exist_ok=True)

    joint_names = list(env.dof_names)
    if robot_name != "go2":
        raise ValueError(f"Deployment config export is not defined for robot: {robot_name}")
    missing = set(joint_names).symmetric_difference(GO2_SDK_JOINT_NAMES)
    if missing:
        raise ValueError(f"Go2 joint names do not match the SDK contract: {sorted(missing)}")

    policy_to_sdk = [GO2_SDK_JOINT_NAMES.index(name) for name in joint_names]
    cfg = {
        "format_version": 1,
        "robot": robot_name,
        "step_dt": float(env.dt),
        "policy": {
            "num_observations": int(env.num_obs),
            "num_actions": int(env.num_actions),
            "joint_names": joint_names,
            "sdk_joint_names": GO2_SDK_JOINT_NAMES,
            "policy_to_sdk": policy_to_sdk,
        },
        "observations": [
            {"name": "base_ang_vel", "size": 3, "scale": float(env.obs_scales.ang_vel)},
            {"name": "projected_gravity", "size": 3, "scale": 1.0},
            {"name": "velocity_commands", "size": 3, "scale": _tensor_list(env.commands_scale)},
            {"name": "joint_pos_rel", "size": env.num_actions, "scale": float(env.obs_scales.dof_pos)},
            {"name": "joint_vel", "size": env.num_actions, "scale": float(env.obs_scales.dof_vel)},
            {"name": "last_action", "size": env.num_actions, "scale": 1.0},
        ],
        "actions": {
            "type": "joint_position",
            "scale": float(env.cfg.control.action_scale),
            "clip": float(env.cfg.normalization.clip_actions),
            "default_joint_pos": _tensor_list(env.default_dof_pos),
        },
        "control": {
            "stiffness": _tensor_list(env.p_gains),
            "damping": _tensor_list(env.d_gains),
            "torque_limits": _tensor_list(env.torque_limits),
        },
        "commands": {
            "lin_vel_x": list(env.cfg.commands.ranges.lin_vel_x),
            "lin_vel_y": list(env.cfg.commands.ranges.lin_vel_y),
            "ang_vel_yaw": list(env.cfg.commands.ranges.ang_vel_yaw),
        },
    }

    output_path = os.path.join(path, "deploy.yaml")
    with open(output_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    return output_path
