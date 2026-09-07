import unittest

import numpy as np

from deploy.common.go2_policy import action_to_target, build_observation, name_to_indices, projected_gravity


POLICY_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
SDK_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


def deploy_cfg():
    default = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]
    return {
        "policy": {"num_observations": 45, "num_actions": 12},
        "observations": [
            {"name": "base_ang_vel", "scale": 0.25},
            {"name": "projected_gravity", "scale": 1.0},
            {"name": "velocity_commands", "scale": [2.0, 2.0, 0.25]},
            {"name": "joint_pos_rel", "scale": 1.0},
            {"name": "joint_vel", "scale": 0.05},
            {"name": "last_action", "scale": 1.0},
        ],
        "actions": {"default_joint_pos": default, "scale": 0.25, "clip": 100.0},
    }


class Go2PolicyInterfaceTest(unittest.TestCase):
    def test_policy_to_sdk_mapping(self):
        self.assertEqual(name_to_indices(SDK_NAMES, POLICY_NAMES).tolist(), [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])

    def test_identity_quaternion_projects_down(self):
        np.testing.assert_allclose(projected_gravity([1, 0, 0, 0]), [0, 0, -1])

    def test_observation_layout_and_action(self):
        cfg = deploy_cfg()
        default = np.asarray(cfg["actions"]["default_joint_pos"], dtype=np.float32)
        obs = build_observation([1, 2, 3], [0, 0, -1], [0.5, -0.5, 1], default, np.ones(12), np.ones(12), cfg)
        self.assertEqual(obs.shape, (45,))
        np.testing.assert_allclose(obs[:9], [0.25, 0.5, 0.75, 0, 0, -1, 1, -1, 0.25])
        np.testing.assert_allclose(obs[9:21], 0)
        np.testing.assert_allclose(obs[21:33], 0.05)
        np.testing.assert_allclose(obs[33:45], 1)
        np.testing.assert_allclose(action_to_target(np.ones(12), cfg), default + 0.25)


if __name__ == "__main__":
    unittest.main()
