import os

import numpy as np
import yaml


def resolve_path(path, root_dir):
    path = path.replace("{LEGGED_GYM_ROOT_DIR}", root_dir)
    return os.path.abspath(path)


def load_deploy_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    validate_policy_config(cfg)
    return cfg


def name_to_indices(source_names, target_names):
    """返回索引，用于从源顺序数组中按目标顺序读取元素。"""
    if len(source_names) != len(set(source_names)):
        raise ValueError("Source joint names must be unique")
    missing = set(target_names) - set(source_names)
    if missing:
        raise ValueError(f"Missing joints: {sorted(missing)}")
    return np.asarray([source_names.index(name) for name in target_names], dtype=np.int64)


def projected_gravity(quaternion_wxyz):
    """将世界坐标系重力方向(0, 0, -1)投影到机身坐标系。"""
    qw, qx, qy, qz = np.asarray(quaternion_wxyz, dtype=np.float32)
    return np.asarray(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def validate_policy_config(cfg):
    """Reject unsupported temporal contracts before loading a policy or opening DDS."""
    if cfg.get("robot") != "go2" or cfg.get("format_version") not in (1, 2):
        raise ValueError("Expected a version 1 or 2 Go2 deployment configuration")
    policy = cfg["policy"]
    him = cfg["format_version"] == 2
    if int(policy["num_actions"]) != 12 or int(policy["num_observations"]) != (270 if him else 45):
        raise ValueError("Invalid Go2 policy dimensions")
    expected = ["base_ang_vel", "projected_gravity", "velocity_commands",
                "joint_pos_rel", "joint_vel", "last_action"]
    if him:
        expected = ["velocity_commands", "base_ang_vel", "projected_gravity",
                    "joint_pos_rel", "joint_vel", "last_action"]
        for key, value in {"architecture": "him", "num_one_step_observations": 45,
                           "history_length": 6, "history_order": "newest_first",
                           "history_reset": "zero_previous_frames",
                           "last_action_semantics": "previous_clipped_policy_action"}.items():
            if policy.get(key) != value:
                raise ValueError(f"Unsupported HIM contract: {key}")
        if not np.isfinite(policy.get("observation_clip", np.nan)) or policy["observation_clip"] <= 0:
            raise ValueError("Invalid observation clip")
    if [item["name"] for item in cfg["observations"]] != expected:
        raise ValueError("Observation order does not match policy architecture")
    if [item["size"] for item in cfg["observations"]] != [3, 3, 3, 12, 12, 12]:
        raise ValueError("Invalid observation component sizes")
    scale = np.asarray(cfg["actions"]["scale"], dtype=np.float32)
    if scale.shape not in ((), (12,)) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid action scale")
    if not np.isfinite(cfg["actions"]["clip"]) or cfg["actions"]["clip"] <= 0:
        raise ValueError("Invalid action clip")
    names = policy["joint_names"]
    if len(names) != 12 or len(set(names)) != 12:
        raise ValueError("Expected twelve unique joints")
    if "sdk_joint_names" in policy:
        mapping = name_to_indices(policy["sdk_joint_names"], names)
        if not np.array_equal(mapping, policy["policy_to_sdk"]):
            raise ValueError("Joint mapping disagrees with joint names")


def build_observation(ang_vel, gravity, command, joint_pos, joint_vel, last_action, cfg):
    """Build one 45-value frame in the declared training order."""
    values = {"base_ang_vel": ang_vel, "projected_gravity": gravity,
              "velocity_commands": command, "joint_pos_rel": np.asarray(joint_pos) -
              np.asarray(cfg["actions"]["default_joint_pos"], dtype=np.float32),
              "joint_vel": joint_vel, "last_action": last_action}
    components = []
    for item in cfg["observations"]:
        value = np.asarray(values[item["name"]], dtype=np.float32)
        if value.shape != (item.get("size", 12 if item["name"] in ("joint_pos_rel", "joint_vel", "last_action") else 3),):
            raise ValueError(f"Invalid observation component: {item['name']}")
        components.append(value * np.asarray(item["scale"], dtype=np.float32))
    obs = np.concatenate(components).astype(np.float32, copy=False)
    if obs.shape != (45,) or not np.isfinite(obs).all():
        raise ValueError(f"Invalid policy observation: shape={obs.shape}, finite={np.isfinite(obs).all()}")
    if cfg.get("format_version") == 2:
        obs = np.clip(obs, -cfg["policy"]["observation_clip"], cfg["policy"]["observation_clip"])
    return obs


class PolicyObservationHistory:
    """One instance per robot; advance exactly once per policy inference."""

    def __init__(self, cfg):
        self.frames = np.zeros((int(cfg["policy"].get("history_length", 1)), 45), dtype=np.float32)

    def reset(self):
        self.frames.fill(0.)

    def append(self, frame):
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (45,) or not np.isfinite(frame).all():
            raise ValueError("Invalid history frame")
        self.frames[1:] = self.frames[:-1].copy()
        self.frames[0] = frame
        return self.frames.reshape(-1).copy()


def action_to_target(action, cfg):
    action_cfg = cfg["actions"]
    action = np.asarray(action, dtype=np.float32)
    expected = int(cfg["policy"]["num_actions"])
    if action.shape != (expected,) or not np.isfinite(action).all():
        raise ValueError(f"Invalid policy action: shape={action.shape}, finite={np.isfinite(action).all()}")
    action = np.clip(action, -float(action_cfg["clip"]), float(action_cfg["clip"]))
    return np.asarray(action_cfg["default_joint_pos"], dtype=np.float32) + action * np.asarray(action_cfg["scale"], dtype=np.float32)
