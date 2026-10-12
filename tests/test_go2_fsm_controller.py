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

    def test_per_joint_velocity_limits_preserve_lower_configuration(self):
        controller = Go2Controller(self.cfg, policy=ZeroPolicy())
        safety = controller.safety
        q = self.cfg.default_joint_pos
        for joint, speed, expected in [(0,22.,True),(1,24.9,True),(2,20.,True),
                                       (0,25.1,False),(2,20.1,False)]:
            dq = np.zeros(12); dq[joint] = speed
            self.assertEqual(safety.check_state(1.,1.,0.,0.,q,dq).safe, expected)
        config = copy.deepcopy(self.cfg); config.data['safety']['max_joint_velocity'] = 12.
        limited = Go2Controller(config, policy=ZeroPolicy())
        np.testing.assert_array_equal(limited.safety.joint_velocity_limits, 12.)

    def test_pd_saturation_precedes_pd_rejection_and_keeps_angle_bounds(self):
        class HipTarget(ZeroPolicy):
            def __call__(self, observation):
                action = torch.zeros((1,12)); action[0,0] = .64
                return action
        controller = Go2Controller(self.cfg, policy=HipTarget(), action_step_mode='limit')
        controller.policy_pd_limit = 14.
        controller.fsm.mode = Go2Mode.POLICY
        q = self.cfg.default_joint_pos.copy(); q[0] = -.9
        state = self.state(q_policy=q)
        command = controller.step(1., state)
        self.assertEqual(command.mode, Go2Mode.POLICY)
        pd = command.kp_sdk*(command.target_pos_sdk-state.joint_pos_sdk)-command.kd_sdk*state.joint_vel_sdk
        self.assertLessEqual(float(np.max(abs(pd))), 14.00001)
        self.assertAlmostEqual(controller.last_action[0], .64)
        # 无可行角度/力矩交集不能用限幅掩盖。
        q = self.cfg.default_joint_pos.copy(); q[0] = controller.safety.joint_upper[0]
        dq = np.zeros(12); dq[0] = 40.
        with self.assertRaisesRegex(ValueError, '无可行目标'):
            controller.safety.limit_pd_target(q,q,dq,self.cfg.stiffness,self.cfg.damping,14.)

    def test_bounded_action_step_preserves_raw_limits_and_applied_history(self):
        class CalfJump(ZeroPolicy):
            def __init__(self): self.observations=[]
            def __call__(self, observation):
                self.observations.append(observation.clone())
                action=torch.zeros((1,12));action[0,5]=1.11
                return action
        policy=CalfJump();controller=Go2Controller(self.cfg,policy=policy,action_step_mode='limit')
        controller.fsm.mode=Go2Mode.POLICY
        result=controller.step(1.,self.state())
        self.assertEqual(result.mode,Go2Mode.POLICY)
        self.assertLessEqual(float(abs(controller.last_action).max()),1.)
        self.assertEqual(controller.last_action_processing['limited_joints'],[5])
        sdk=self.cfg.policy_to_sdk[5]
        self.assertAlmostEqual(result.target_pos_sdk[sdk],self.cfg.default_joint_pos[5]+.25,places=6)
        controller.step(1.02,self.state(timestamp=1.02))
        self.assertAlmostEqual(policy.observations[1][0,-12+5].item(),1.)
        self.assertAlmostEqual(controller.last_action[5],1.11,places=6)
        # NaN或原始幅值越界不能通过步长限幅隐藏。
        for value in (float('nan'),float('inf'),4.1):
            class Invalid(ZeroPolicy):
                def __call__(self, observation):return torch.full((1,12),value)
            invalid=Go2Controller(self.cfg,policy=Invalid(),action_step_mode='limit')
            invalid.fsm.mode=Go2Mode.POLICY
            self.assertEqual(invalid.step(1.,self.state()).mode,Go2Mode.DAMPING)
            np.testing.assert_array_equal(invalid.last_action,0.)
        # 关节速度、姿态与通信保护仍优先于策略处理。
        for field,value in [('joint_vel_sdk',np.full(12,30.)),('quaternion_wxyz',np.array([.7071,.7071,0.,0.])),('timestamp',0.)]:
            invalid=Go2Controller(self.cfg,policy=policy,action_step_mode='limit')
            invalid.fsm.mode=Go2Mode.POLICY
            state=self.state();setattr(state,field,value)
            self.assertEqual(invalid.step(1.,state).mode,Go2Mode.DAMPING)

    def test_limited_action_history_can_be_replayed_from_logs(self):
        from unittest.mock import patch
        from deploy.deploy_real import go2_policy_replay
        class GrowingCalf(ZeroPolicy):
            def __call__(self, observation):
                action=observation[:,-12:].clone();action[:,5]+=1.1
                return action
        controller=Go2Controller(self.cfg,policy=GrowingCalf(),action_step_mode='limit')
        controller.action_step_limit=.64;controller.fsm.mode=Go2Mode.POLICY
        controller.policy_pd_limit=14.
        rows=[]
        for i in range(3):
            now=8.+i*.02
            q=self.cfg.default_joint_pos.copy();q[5]=-2.5
            state=self.state(timestamp=now,q_policy=q)
            command=controller.step(now,state)
            self.assertEqual(command.mode,Go2Mode.POLICY)
            rows.append({'mode':'policy','reason':'','time':now,'state_time':now,
                         'quaternion_wxyz':state.quaternion_wxyz.tolist(),'angular_velocity':state.angular_velocity.tolist(),
                         'joint_pos_sdk':state.joint_pos_sdk.tolist(),'joint_vel_sdk':state.joint_vel_sdk.tolist(),
                         'velocity_command':[0,0,0],'target_pos_sdk':command.target_pos_sdk.tolist(),
                         'feedback':{'policy_inference_count':i+1,'test_elapsed':now,'policy_entry_blend_seconds':.5,
                                     'policy_limits_version':'per_joint_v1', 'policy_pd_limit_nm':14.,'policy_action_processing':controller.last_action_processing}})
        def make_controller(config,action_step_mode):
            return Go2Controller(config,policy=GrowingCalf(),action_step_mode=action_step_mode)
        with patch.object(go2_policy_replay,'Go2Controller',side_effect=make_controller):
            result=go2_policy_replay.replay(rows,self.cfg)
            self.assertTrue(result['historical_outputs_match'])
            self.assertEqual(result['inferences_replayed'],3)
            bad=copy.deepcopy(rows);bad[1]['feedback']['policy_action_processing']['step_limit']=2.
            with self.assertRaisesRegex(ValueError,'阈值无效'):go2_policy_replay.replay(bad,self.cfg)

    def test_action_limiter_float_rounding_never_exceeds_existing_delta(self):
        controller=Go2Controller(self.cfg,policy=ZeroPolicy(),action_step_mode='limit')
        rng=np.random.RandomState(7)
        for _ in range(1000):
            previous=rng.uniform(-4,4,12).astype(np.float32)
            controller.last_action=previous
            raw=rng.uniform(-4,4,12).astype(np.float32)
            applied=controller.process_policy_action(raw)
            self.assertTrue(np.all(abs(applied-previous)<=controller.safety.max_action_delta))
            self.assertTrue(np.all(abs(applied)<=controller.safety.max_action))
            self.assertTrue(np.all(abs(applied-raw)<=abs(previous-raw)+1e-6))

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
