import os
import copy
from dataclasses import replace
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

    def test_rejected_target_records_joint_and_keeps_damping(self):
        class OutOfRangePolicy(ZeroPolicy):
            def __call__(self, observation):
                action=torch.zeros((1,12));action[0,11]=2.8
                return action
        data = copy.deepcopy(self.cfg.data)
        data['safety']['policy_target_mode'] = 'reject'
        controller=Go2Controller(replace(self.cfg,data=data),policy=OutOfRangePolicy())
        controller.request_fix_stand(1.,self.state())
        end=1.+float(self.cfg.data['fsm']['stand_duration'])
        controller.last_step_time=end-.02
        controller.step(end,self.state(timestamp=end))
        self.assertTrue(controller.request_policy(end))
        controller.last_action[11]=2.5
        result=controller.step(end+.02,self.state(timestamp=end+.02))
        self.assertEqual(result.mode,Go2Mode.DAMPING)
        self.assertEqual(result.reason,'目标关节角超过安全限制')
        fault=controller.last_policy_fault
        self.assertEqual(fault['violating_joints'][0]['joint'],'RR_calf_joint')
        self.assertGreater(fault['violating_joints'][0]['target_rad'],fault['violating_joints'][0]['upper_rad'])
        self.assertEqual(len(fault['observation']),45)
        self.assertAlmostEqual(controller.last_action[11],2.5)
        again=controller.step(end+.04,self.state(timestamp=end+.04))
        self.assertEqual(again.mode,Go2Mode.DAMPING)
        self.assertIs(controller.last_policy_fault,fault)

    def test_known_rr_overrun_is_clipped_but_raw_action_history_is_preserved(self):
        class KnownFaultPolicy(ZeroPolicy):
            def __call__(self, observation):
                action = torch.zeros((1,12)); action[0,11] = 2.7759674
                return action
        controller = Go2Controller(self.cfg, policy=KnownFaultPolicy())
        controller.request_fix_stand(1., self.state())
        end = 1. + float(self.cfg.data['fsm']['stand_duration'])
        controller.last_step_time = end - .02
        controller.step(end, self.state(timestamp=end))
        controller.request_policy(end)
        controller.last_action[11] = 2.5
        command = controller.step(end+.02, self.state(timestamp=end+.02))
        self.assertEqual(command.mode, Go2Mode.POLICY)
        sdk_joint = self.cfg.policy_to_sdk[11]
        self.assertAlmostEqual(command.target_pos_sdk[sdk_joint], self.cfg.safety['joint_upper'][11], places=5)
        self.assertAlmostEqual(controller.last_action[11], 2.7759674, places=5)
        processing = controller.last_target_processing
        self.assertEqual(processing['clipped_joints'], [11])
        self.assertAlmostEqual(processing['max_correction_rad'], .03175185, places=5)
        # 实际关节越界仍停止，不能由参考目标限幅隐藏。
        q = self.cfg.default_joint_pos.copy(); q[11] = -.7
        stopped = controller.step(end+.04, self.state(timestamp=end+.04, q_policy=q))
        self.assertEqual(stopped.mode, Go2Mode.DAMPING)
        self.assertIn('关节位置', stopped.reason)

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
