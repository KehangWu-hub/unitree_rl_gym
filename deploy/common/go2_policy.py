import os

import numpy as np
import yaml


def resolve_path(path, root_dir):
    path = path.replace("{LEGGED_GYM_ROOT_DIR}", root_dir)
    return os.path.abspath(path)


def load_deploy_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if cfg.get("format_version") != 1 or cfg.get("robot") != "go2":
        raise ValueError("Expected a version 1 Go2 deployment configuration")
    return cfg


def name_to_indices(source_names, target_names):
    """Return indices that read target order from an array in source order."""
    if len(source_names) != len(set(source_names)):
        raise ValueError("Source joint names must be unique")
    missing = set(target_names) - set(source_names)
    if missing:
        raise ValueError(f"Missing joints: {sorted(missing)}")
    return np.asarray([source_names.index(name) for name in target_names], dtype=np.int64)


def projected_gravity(quaternion_wxyz):
    """Project world gravity (0, 0, -1) into the body frame."""
    qw, qx, qy, qz = np.asarray(quaternion_wxyz, dtype=np.float32)
    return np.asarray(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def build_observation(ang_vel, gravity, command, joint_pos, joint_vel, last_action, cfg):
    """Build the exact 45-dimensional Go2 actor observation."""
    scales = {item["name"]: item["scale"] for item in cfg["observations"]}
    default_pos = np.asarray(cfg["actions"]["default_joint_pos"], dtype=np.float32)
    obs = np.concatenate(
        (
            np.asarray(ang_vel, dtype=np.float32) * scales["base_ang_vel"],
            np.asarray(gravity, dtype=np.float32) * scales["projected_gravity"],
            np.asarray(command, dtype=np.float32) * np.asarray(scales["velocity_commands"], dtype=np.float32),
            (np.asarray(joint_pos, dtype=np.float32) - default_pos) * scales["joint_pos_rel"],
            np.asarray(joint_vel, dtype=np.float32) * scales["joint_vel"],
            np.asarray(last_action, dtype=np.float32) * scales["last_action"],
        )
    ).astype(np.float32, copy=False)
    expected = int(cfg["policy"]["num_observations"])
    if obs.shape != (expected,) or not np.isfinite(obs).all():
        raise ValueError(f"Invalid policy observation: shape={obs.shape}, finite={np.isfinite(obs).all()}")
    return obs


def action_to_target(action, cfg):
    action_cfg = cfg["actions"]
    action = np.asarray(action, dtype=np.float32)
    expected = int(cfg["policy"]["num_actions"])
    if action.shape != (expected,) or not np.isfinite(action).all():
        raise ValueError(f"Invalid policy action: shape={action.shape}, finite={np.isfinite(action).all()}")
    action = np.clip(action, -float(action_cfg["clip"]), float(action_cfg["clip"]))
    return np.asarray(action_cfg["default_joint_pos"], dtype=np.float32) + action * float(action_cfg["scale"])
