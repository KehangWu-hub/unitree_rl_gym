import os
import unittest

import numpy as np
import torch

from deploy.deploy_real.deploy_real_go2 import Go2Controller, Go2FSM, Go2Mode, Go2RealConfig, Go2State


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_CONFIG = os.path.join(ROOT, "deploy", "deploy_real", "configs", "go2.yaml")


class ZeroPolicy:
    def eval(self):
        return self

    def __call__(self, observation):
        return torch.zeros((observation.shape[0], 12), dtype=torch.float32)


class Go2FSMControllerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Go2RealConfig.load(RUNTIME_CONFIG, ROOT)

    def state(self, timestamp=1.0, q_policy=None):
        q_policy = self.cfg.default_joint_pos if q_policy is None else np.asarray(q_policy, dtype=np.float32)
        q_sdk = np.empty(12, dtype=np.float32)
        q_sdk[self.cfg.policy_to_sdk] = q_policy
        return Go2State(timestamp, np.array([1, 0, 0, 0]), np.zeros(3), q_sdk, np.zeros(12))

    def test_fsm_requires_completed_stand_before_policy(self):
        duration = float(self.cfg.data["fsm"]["stand_duration"])
        fsm = Go2FSM(self.cfg.default_joint_pos, duration)
        self.assertFalse(fsm.enter_policy(0.0))
        self.assertTrue(fsm.enter_fix_stand(0.0, np.zeros(12)))
        fsm.stand_target(duration - 0.1)
        self.assertFalse(fsm.enter_policy(duration - 0.1))
        np.testing.assert_allclose(fsm.stand_target(duration), self.cfg.default_joint_pos)
        self.assertTrue(fsm.enter_policy(duration))

    def test_controller_runs_passive_stand_and_policy_in_sdk_order(self):
        controller = Go2Controller(self.cfg, policy=ZeroPolicy())
        state = self.state()
        passive = controller.step(1.0, state)
        self.assertEqual(passive.mode, Go2Mode.PASSIVE)
        np.testing.assert_allclose(passive.kp_sdk, 0)

        self.assertTrue(controller.request_fix_stand(1.0, state))
        end = 1.0 + float(self.cfg.data["fsm"]["stand_duration"])
        controller.last_step_time = end - 0.02
        stand = controller.step(end, self.state(timestamp=end))
        self.assertEqual(stand.mode, Go2Mode.FIX_STAND)
        self.assertTrue(controller.request_policy(end))
        policy = controller.step(end + 0.02, self.state(timestamp=end + 0.02))
        self.assertEqual(policy.mode, Go2Mode.POLICY)
        expected = np.empty(12, dtype=np.float32)
        expected[self.cfg.policy_to_sdk] = self.cfg.default_joint_pos
        np.testing.assert_allclose(policy.target_pos_sdk, expected)

    def test_stale_state_and_policy_jump_enter_damping(self):
        controller = Go2Controller(self.cfg, policy=ZeroPolicy())
        result = controller.step(2.0, self.state(timestamp=1.0))
        self.assertEqual(result.mode, Go2Mode.DAMPING)
        self.assertIn("通信超时", result.reason)

        class JumpPolicy(ZeroPolicy):
            def __call__(self, observation):
                return torch.full((1, 12), 2.0)

        controller = Go2Controller(self.cfg, policy=JumpPolicy())
        state = self.state()
        controller.request_fix_stand(1.0, state)
        end = 1.0 + float(self.cfg.data["fsm"]["stand_duration"])
        controller.last_step_time = end - 0.02
        controller.step(end, self.state(timestamp=end))
        controller.request_policy(end)
        result = controller.step(end + 0.02, self.state(timestamp=end + 0.02))
        self.assertEqual(result.mode, Go2Mode.DAMPING)
        self.assertIn("单步变化", result.reason)


if __name__ == "__main__":
    unittest.main()
