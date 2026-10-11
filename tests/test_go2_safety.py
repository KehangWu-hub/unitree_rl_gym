import unittest

import numpy as np

from deploy.deploy_real.deploy_real_go2 import Go2Safety


class Go2SafetyTest(unittest.TestCase):
    def setUp(self):
        limits = {
            "state_timeout": 0.1, "max_loop_dt": 0.05,
            "max_roll": 0.8, "max_pitch": 0.8,
            "max_joint_velocity": 25.0, "max_action": 4.0,
            "max_action_delta": 1.0, "torque_limit_ratio": 0.8,
            "state_joint_margin": 0.02,
            "joint_lower": [-1, -0.5, -2.7] * 4,
            "joint_upper": [1, 4.5, -0.8] * 4,
        }
        self.safety = Go2Safety(limits, np.full(12, 20), np.full(12, 0.5), np.full(12, 40))
        self.q = np.asarray([0, 0.8, -1.5] * 4, dtype=np.float32)

    def test_runtime_joint_limits_match_training_urdf(self):
        import xml.etree.ElementTree as ET
        from pathlib import Path
        from deploy.deploy_real.deploy_real_go2 import Go2RealConfig
        root=Path(__file__).resolve().parents[1]
        cfg=Go2RealConfig.load(root/'deploy/deploy_real/configs/go2.yaml',str(root))
        limits={j.attrib['name']:j.find('limit').attrib for j in ET.parse(root/'resources/robots/go2/urdf/go2.urdf').getroot().findall('joint') if j.find('limit') is not None}
        for i,name in enumerate(cfg.data['policy']['joint_names']):
            self.assertAlmostEqual(cfg.safety['joint_lower'][i],float(limits[name]['lower']),places=5)
            self.assertAlmostEqual(cfg.safety['joint_upper'][i],float(limits[name]['upper']),places=5)

    def test_nominal_state_and_command_pass(self):
        self.assertTrue(self.safety.check_state(1.0, 0.95, 0.1, -0.1, self.q, np.zeros(12), 0.02).safe)
        target = self.q + 0.1
        self.assertTrue(self.safety.check_policy_command(np.full(12, 0.4), np.zeros(12), target, self.q, np.zeros(12)).safe)

    def test_timeout_tilt_and_velocity_fail(self):
        self.assertIn("通信超时", self.safety.check_state(1.0, 0.8, 0, 0, self.q, np.zeros(12)).reason)
        self.assertIn("姿态", self.safety.check_state(1.0, 0.95, 0.9, 0, self.q, np.zeros(12)).reason)
        self.assertIn("关节速度", self.safety.check_state(1.0, 0.95, 0, 0, self.q, np.full(12, 30)).reason)
        near_limit = self.q.copy()
        near_limit[2] = -2.71
        self.assertTrue(self.safety.check_state(1.0, 0.95, 0, 0, near_limit, np.zeros(12)).safe)

    def test_action_jump_target_and_torque_fail(self):
        action = np.zeros(12)
        action[0] = 1.1
        self.assertIn("单步变化", self.safety.check_policy_command(action, np.zeros(12), self.q, self.q, np.zeros(12)).reason)
        target = self.q.copy()
        target[2] = -0.7
        self.assertIn("目标关节角", self.safety.check_policy_command(np.zeros(12), np.zeros(12), target, self.q, np.zeros(12)).reason)
        target = self.q.copy()
        target[0] = 0.9
        low_torque_safety = Go2Safety({**self.safety_config(), "torque_limit_ratio": 0.1}, np.full(12, 20), np.full(12, 0.5), np.full(12, 40))
        self.assertIn("PD力矩", low_torque_safety.check_policy_command(np.zeros(12), np.zeros(12), target, self.q, np.zeros(12)).reason)

    def test_non_finite_values_are_rejected(self):
        bad = self.q.copy()
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            self.safety.check_state(1.0, 0.95, 0, 0, bad, np.zeros(12))

    @staticmethod
    def safety_config():
        return {
            "state_timeout": 0.1, "max_loop_dt": 0.05,
            "max_roll": 0.8, "max_pitch": 0.8,
            "max_joint_velocity": 25.0, "max_action": 4.0,
            "max_action_delta": 1.0, "torque_limit_ratio": 0.8,
            "state_joint_margin": 0.02,
            "joint_lower": [-1, -0.5, -2.7] * 4,
            "joint_upper": [1, 4.5, -0.8] * 4,
        }


if __name__ == "__main__":
    unittest.main()
