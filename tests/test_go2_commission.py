import copy
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
import threading
import time
import os
from unittest.mock import Mock, patch

import numpy as np

from deploy.deploy_real.deploy_real_go2 import Go2Mode, Go2RealConfig, Go2State
from deploy.deploy_real.go2_commission import ProneTest, StandCheck, PolicyCheck, VerifiedTransport, bounded_control_gc, check_prone, check_resting_for_restore, parse_args, release_mode, startup_lease, serve, assess_response, assess_stand, assess_policy
from deploy.deploy_real.go2_watchdog import request


ROOT = Path(__file__).resolve().parents[1]


class Go2CommissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Go2RealConfig.load(ROOT / 'deploy/deploy_real/configs/go2.yaml', str(ROOT))
        # 既有用例验证10.5/12Nm历史日志兼容；当前默认单独验证。
        cls.cfg.data.pop('commissioning', None)

    def test_current_default_uses_verified_torque_profile(self):
        cfg = Go2RealConfig.load(ROOT / 'deploy/deploy_real/configs/go2.yaml', str(ROOT))
        with tempfile.NamedTemporaryFile(mode='w') as log, patch('deploy.deploy_real.go2_commission.assess_stand',return_value={'result':'stand_hold_lower_passed'}):
            trial = PolicyCheck(cfg, log.name)
        self.assertEqual(trial.policy_pd_limit, 14.)
        self.assertEqual(trial.torque_limit, 16.)

    def prone(self, now=1.0):
        # 来源：2026-10-09实体趴卧状态；SDK顺序FR、FL、RR、RL。
        q = np.array([-.0554, 1.2737, -2.8184, .0422, 1.270, -2.8053,
                      -.3279, 1.2945, -2.7995, .3667, 1.3079, -2.8213])
        return Go2State(now, np.array([1., 0, 0, 0]), np.zeros(3), q, np.zeros(12))

    def test_policy_requires_successful_stand_log(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl') as log:
            log.write('{"reason":"fault"}\n'); log.flush()
            with self.assertRaisesRegex(ValueError, '前置站立验收未通过'):
                PolicyCheck(self.cfg, log.name)

    def test_forward_rejects_prior_zero_with_different_actuation(self):
        with tempfile.NamedTemporaryFile(mode='w') as log, patch('deploy.deploy_real.go2_commission.assess_policy',return_value={'result':'zero_velocity_policy_trial_passed'}):
            log.write('{"mode":"policy","feedback":{}}'); log.flush()
            with self.assertRaisesRegex(ValueError,'当前接管曲线和PD限幅'):
                PolicyCheck(self.cfg,log.name,forward=True)

    def test_policy_hold_gate_rate_and_stop(self):
        import torch
        class Actor:
            def __init__(self): self.calls = 0
            def __call__(self, obs):
                self.calls += 1
                return torch.zeros((1, 12))
        with tempfile.NamedTemporaryFile(mode='w') as log, patch('deploy.deploy_real.go2_commission.assess_stand', return_value={'result': 'stand_hold_lower_passed'}):
            actor = Actor()
            test = PolicyCheck(self.cfg, log.name, policy=actor)
            def state(now, tick):
                q = self.cfg.default_joint_pos[self.cfg.sdk_to_policy].copy()
                s = Go2State(now, np.array([1.,0,0,0]), np.zeros(3), q, np.zeros(12))
                s.feedback = {'joint_tau_est_sdk': [0.]*12, 'state_tick': tick}
                return s
            test.arm(1., state(1., 0))
            for tick in range(4500):
                now = 1 + tick * .002
                cmd = test.output(now, state(now, tick))
                self.assertEqual(cmd.mode, Go2Mode.POLICY if tick >= 3500 else Go2Mode.FIX_STAND)
            self.assertEqual(test.inference_count, 100)
            self.assertEqual(actor.calls, 101)  # 一次预热加100次策略计算。
            np.testing.assert_array_equal(test.controller.command, np.zeros(3))
            cmd = test.output(10., state(10., 4500))
            self.assertEqual(cmd.mode, Go2Mode.FIX_STAND)
            test.stop(10.002, '第二终端请求停止')
            self.assertEqual(test.output(10.002, state(10.002,4501)).mode, Go2Mode.DAMPING)
            np.testing.assert_array_equal(test.output(10.004,state(10.004,4502)).kp_sdk,0)
            test = PolicyCheck(self.cfg, log.name, policy=Actor())
            test.arm(1., state(1.,0))
            test.last_time = 7.998
            cmd = test.output(8., state(8.,1))
            self.assertEqual(cmd.mode, Go2Mode.DAMPING)
            self.assertIn('稳定保持不足', test.reason)

    def test_policy_entry_is_continuous_pd_blend(self):
        from deploy.deploy_real.deploy_real_go2 import Go2Command
        with tempfile.NamedTemporaryFile(mode='w') as log, patch('deploy.deploy_real.go2_commission.assess_stand',return_value={'result':'stand_hold_lower_passed'}):
            import torch
            test=PolicyCheck(self.cfg,log.name,policy=lambda obs:torch.zeros((1,12)))
        test.started=1.
        command=Go2Command(Go2Mode.POLICY,test.stand+.1,np.full(12,20.),np.full(12,.5))
        q=test.stand-.05; dq=np.full(12,.2)
        for offset in (0.,.25,.5):
            out=test.blend_policy_entry(8.+offset,command)
            ratio=offset/.5; ratio=ratio*ratio*(3-2*ratio)
            old_pd=test.hold_kp*(test.stand-q)-test.hold_kd*dq
            new_pd=command.kp_sdk*(command.target_pos_sdk-q)-command.kd_sdk*dq
            np.testing.assert_allclose(out.kp_sdk*(out.target_pos_sdk-q)-out.kd_sdk*dq,old_pd*(1-ratio)+new_pd*ratio,atol=1e-5)
        np.testing.assert_allclose(test.blend_policy_entry(8.,command).target_pos_sdk,test.stand)
        np.testing.assert_allclose(test.blend_policy_entry(8.5,command).target_pos_sdk,command.target_pos_sdk)

    def test_policy_pd_cap_retains_model_limits_and_gains(self):
        from deploy.deploy_real.deploy_real_go2 import Go2Command
        import torch
        with tempfile.NamedTemporaryFile(mode='w') as log, patch('deploy.deploy_real.go2_commission.assess_stand',return_value={'result':'stand_hold_lower_passed'}):
            test=PolicyCheck(self.cfg,log.name,policy=lambda obs:torch.zeros((1,12)))
        state=Go2State(1.,np.array([1.,0,0,0]),np.zeros(3),test.stand.copy(),np.full(12,.2))
        for sign in (-1,1):
            cmd=Go2Command(Go2Mode.POLICY,test.stand+sign*.6,np.full(12,20.),np.full(12,.5))
            out=test.limit_policy_pd(cmd,state)
            pd=out.kp_sdk*(out.target_pos_sdk-state.joint_pos_sdk)-out.kd_sdk*state.joint_vel_sdk
            np.testing.assert_allclose(pd,sign*10.5,atol=1e-5)
            np.testing.assert_array_equal(out.kp_sdk,cmd.kp_sdk)
            np.testing.assert_array_equal(out.kd_sdk,cmd.kd_sdk)
            self.assertTrue(np.all(out.target_pos_sdk>=test.lower) and np.all(out.target_pos_sdk<=test.upper))
        cmd.target_pos_sdk=test.stand+.1
        np.testing.assert_allclose(test.limit_policy_pd(cmd,state).target_pos_sdk,cmd.target_pos_sdk)

    def test_candidate_torque_settings_are_bounded_and_need_new_zero_trial(self):
        from dataclasses import replace
        import json
        for pd, feedback, valid in [(14.,16.,True),(0.,12.,False),(14.,12.,False),(20.,22.,False),(float('nan'),16.,False)]:
            data = copy.deepcopy(self.cfg.data)
            data['commissioning'] = {'policy_pd_limit_nm':pd, 'feedback_torque_limit_nm':feedback}
            cfg = replace(self.cfg,data=data)
            with tempfile.NamedTemporaryFile(mode='w') as log, patch('deploy.deploy_real.go2_commission.assess_stand',return_value={'result':'stand_hold_lower_passed'}):
                if valid:
                    trial = PolicyCheck(cfg,log.name)
                    self.assertEqual(trial.policy_pd_limit,14.)
                    self.assertEqual(trial.torque_limit,16.)
                else:
                    with self.assertRaisesRegex(ValueError,'验收力矩参数'):PolicyCheck(cfg,log.name)
            if valid:
                with tempfile.NamedTemporaryFile(mode='w') as log, patch('deploy.deploy_real.go2_commission.assess_policy',return_value={'result':'zero_velocity_policy_trial_passed'}):
                    log.write(json.dumps({'mode':'policy','feedback':{'policy_entry_blend_seconds':.5,'policy_pd_limit_nm':10.5}}));log.flush()
                    with self.assertRaisesRegex(ValueError,'当前接管曲线和PD限幅'):PolicyCheck(cfg,log.name,forward=True)

    def test_forward_timeline_zeroing_exit_and_log(self):
        import torch
        import json
        for speed in (.1,.3):
            actor = lambda obs: torch.zeros((1, 12))
            with tempfile.NamedTemporaryFile(mode='w') as log, patch('deploy.deploy_real.go2_commission.assess_policy', return_value={'result': 'zero_velocity_policy_trial_passed'}):
                log.write('{"mode":"policy","feedback":{"policy_entry_blend_seconds":0.5,"policy_pd_limit_nm":10.5}}'); log.flush()
                test = PolicyCheck(self.cfg, log.name, policy=actor, forward=True, forward_speed=speed)
            def state(now, tick, q):
                s = Go2State(now, np.array([1.,0,0,0]), np.zeros(3), q.copy(), np.zeros(12))
                s.feedback = {'joint_tau_est_sdk': [0.]*12, 'state_tick': tick}
                return s
            nominal = self.cfg.default_joint_pos[self.cfg.sdk_to_policy].copy()
            test.arm(1., state(1., 0, nominal))
            rows = []
            for tick in range(9501):
                now = 1 + tick * .002
                phase = test.phase(now)[0]
                q = nominal if phase == 'policy' or (phase == 'lower' and test.exit_start is None) else test.profile(now)[0]
                st = state(now, tick, q)
                cmd = test.output(now, st)
                self.assertEqual(cmd.mode, Go2Mode.DAMPING if tick == 9500 else Go2Mode.POLICY if phase == 'policy' else Go2Mode.FIX_STAND)
                velocity = test.controller.command.copy()
                self.assertTrue(0 <= velocity[0] <= speed + .000001)
                np.testing.assert_array_equal(velocity[1:],0)
                if now - 1 >= 10.02:
                    np.testing.assert_array_equal(velocity,0)
                rows.append({'time': now, 'mode': cmd.mode.value, 'reason': cmd.reason,
                             'velocity_command': velocity.tolist(), 'kp_sdk': cmd.kp_sdk.tolist(), 'kd_sdk': cmd.kd_sdk.tolist(),
                             'target_pos_sdk': cmd.target_pos_sdk.tolist(),
                             'joint_pos_sdk': st.joint_pos_sdk.tolist(), 'joint_vel_sdk': st.joint_vel_sdk.tolist(),
                             'quaternion_wxyz': st.quaternion_wxyz.tolist(),
                             'feedback': {'test_elapsed': now-1, 'stand_profile': 'forward19', 'policy_inference_count': test.inference_count, 'policy_entry_blend_seconds': test.entry_blend_seconds, 'policy_pd_limit_nm': test.policy_pd_limit, 'forward_speed_mps': speed}})
            self.assertEqual(test.inference_count,250)
            final = dict(rows[-1],time=21.,joint_pos_sdk=self.prone().joint_pos_sdk.tolist())
            rows.append(final)
            self.assertEqual(assess_policy(rows,self.cfg)['result'],'forward_policy_trial_log_passed',assess_policy(rows,self.cfg))
            self.assertFalse(assess_policy(rows,self.cfg)['physical_translation_and_stop_confirmed'])
            # 缺少归零观察与超时延长不能被算作通过。
            bad = copy.deepcopy(rows)
            for r in bad:
                if r['mode']=='policy' and r['feedback']['test_elapsed']>=10.1:
                    r['velocity_command']=[.1,0,0]
            self.assertEqual(assess_policy(bad,self.cfg)['result'],'inconclusive')
            bad[-1]['reason']='第二终端请求停止'
            self.assertEqual(assess_policy(bad,self.cfg)['result'],'inconclusive')
            with tempfile.NamedTemporaryFile(mode='w') as log:
                log.write(json.dumps(rows[0])); log.flush()
                with self.assertRaisesRegex(ValueError,'零速度策略验收未通过'):
                    PolicyCheck(self.cfg,log.name,policy=actor,forward=True)

    def test_restore_accepts_settled_emergency_pose_but_never_active_test(self):
        s=self.prone()
        s.joint_pos_sdk=np.array([-.383,1.063,-2.627,.387,1.052,-2.598,-.499,1.177,-2.683,.609,1.124,-2.632])
        check_resting_for_restore(self.cfg,s,1.)
        with self.assertRaises(ValueError): check_prone(self.cfg,s,1.)
        s.joint_pos_sdk[:]=self.cfg.default_joint_pos[self.cfg.sdk_to_policy]
        with self.assertRaises(ValueError): check_resting_for_restore(self.cfg,s,1.)
        s=self.prone(); s.joint_vel_sdk[0]=.3
        with self.assertRaises(ValueError): check_resting_for_restore(self.cfg,s,1.)
        with self.assertRaises(ValueError): check_resting_for_restore(self.cfg,self.prone(),1.2)

    def test_motion_velocity_uses_model_limits_while_stationary_checks_remain_strict(self):
        import torch
        for forward, speed, dq, expected in ((True,.3,7.5,Go2Mode.POLICY),(True,.3,8.1,Go2Mode.POLICY),(True,.1,2.1,Go2Mode.POLICY),(True,.3,20.1,Go2Mode.DAMPING),(False,.3,2.1,Go2Mode.DAMPING)):
            gate='assess_policy' if forward else 'assess_stand'
            result='zero_velocity_policy_trial_passed' if forward else 'stand_hold_lower_passed'
            with tempfile.NamedTemporaryFile(mode='w') as log, patch('deploy.deploy_real.go2_commission.'+gate,return_value={'result':result}):
                log.write('{"mode":"policy","feedback":{"policy_entry_blend_seconds":0.5,"policy_pd_limit_nm":10.5}}'); log.flush()
                test=PolicyCheck(self.cfg,log.name,policy=lambda obs:torch.zeros((1,12)),forward=forward,forward_speed=speed)
            q=test.stand.copy()
            st=Go2State(1.,np.array([1.,0,0,0]),np.zeros(3),q,np.zeros(12));st.feedback={'joint_tau_est_sdk':[0.]*12,'state_tick':0}
            test.arm(1.,st)
            test.hold_samples=50; test.hold_first=7.;test.hold_last=7.9;test.last_time=8.098
            st.timestamp=8.1;st.joint_vel_sdk[:]=dq
            self.assertEqual(test.output(8.1,st).mode,expected)
        test=StandCheck(self.cfg,start_standing=True)
        st.timestamp=1.;st.joint_vel_sdk[:]=0
        test.arm(1.,st)
        st.timestamp=1.002;st.joint_vel_sdk[:]=2.1
        self.assertEqual(test.output(1.002,st).mode,Go2Mode.DAMPING)

    def test_handoff_clock_starts_after_release_without_rechecking_stillness(self):
        import torch
        with tempfile.NamedTemporaryFile(mode='w') as log, patch('deploy.deploy_real.go2_commission.assess_stand',return_value={'result':'stand_hold_lower_passed'}):
            test=PolicyCheck(self.cfg,log.name,policy=lambda obs:torch.zeros((1,12)))
        st=Go2State(1.,np.array([1.,0,0,0]),np.zeros(3),test.stand.copy(),np.zeros(12))
        st.feedback={'joint_tau_est_sdk':[0.]*12,'state_tick':0}
        test.arm(1.,st)
        initial=test.initial.copy()
        test.begin_handoff(1.25)
        self.assertAlmostEqual(test.handoff_delay_seconds,.25)
        self.assertEqual(test.started,1.25)
        self.assertEqual(test.controller.fsm.mode_started_at,1.25)
        np.testing.assert_array_equal(test.initial,initial)
        st.timestamp=1.25;st.joint_vel_sdk[:]=.4
        cmd=test.output(1.25,st)
        self.assertEqual(cmd.mode,Go2Mode.FIX_STAND)
        np.testing.assert_allclose(cmd.target_pos_sdk,initial)
        with self.assertRaises(ValueError):test.begin_handoff(1.3)

    def test_keyboard_input_timeout_zero_and_budget(self):
        import torch
        with tempfile.NamedTemporaryFile(mode='w') as log, patch('deploy.deploy_real.go2_commission.assess_policy',return_value={'result':'zero_velocity_policy_trial_passed'}):
            log.write('{"mode":"policy","feedback":{"policy_entry_blend_seconds":0.5,"policy_pd_limit_nm":10.5}}');log.flush()
            test=PolicyCheck(self.cfg,log.name,policy=lambda obs:torch.zeros((1,12)),forward=True,forward_speed=.3,keyboard=True)
        test.started=1.
        test.feed_key('w',8.9);self.assertFalse(test.keyboard_event['accepted'])
        np.testing.assert_array_equal(test.velocity_profile(9.),0)
        test.feed_key('w',9.)
        for i in range(1,11):test.velocity_profile(9.+i*.02)
        self.assertGreater(test.keyboard_velocity,.19)
        test.feed_key('x',9.201)
        np.testing.assert_array_equal(test.velocity_profile(9.22),0)
        test.feed_key('w',9.3);test.velocity_profile(9.32)
        np.testing.assert_array_equal(test.velocity_profile(9.56),0)
        for i in range(200):
            now=9.6+i*.02;test.feed_key('w',now);test.velocity_profile(now)
        self.assertTrue(2<=test.keyboard_used<=2.03)
        self.assertEqual(test.keyboard_velocity,0)
        np.testing.assert_array_equal(test.velocity_profile(15.),0)

    def test_keyboard_lateral_yaw_isolation_and_signed_budget(self):
        import torch
        for axis,key,opposite,index,limit in [('lateral','d','a',1,.30),('yaw','e','q',2,.3)]:
            with self.subTest(axis=axis), tempfile.NamedTemporaryFile(mode='w') as log, patch('deploy.deploy_real.go2_commission.assess_policy',return_value={'result':'zero_velocity_policy_trial_passed'}):
                log.write('{"mode":"policy","feedback":{"policy_entry_blend_seconds":0.5,"policy_pd_limit_nm":10.5}}');log.flush()
                test=PolicyCheck(self.cfg,log.name,policy=lambda obs:torch.zeros((1,12)),forward=True,forward_speed=.4,keyboard=True,keyboard_axis=axis)
            test.started=1.;test.velocity_profile(9.)
            test.feed_key('w',9.);self.assertIsNone(test.keyboard_event)
            test.feed_key(key,9.)
            for i in range(1,21): v=test.velocity_profile(9.+i*.02)
            self.assertAlmostEqual(v[index],-limit,places=5)
            np.testing.assert_array_equal(np.delete(v,index),0)
            self.assertGreater(test.keyboard_used,.35)
            test.feed_key(opposite,9.41)
            np.testing.assert_array_equal(test.velocity_profile(9.42),0)
            self.assertGreater(test.velocity_profile(9.44)[index],0)
            test.feed_key('x',9.45)
            np.testing.assert_array_equal(test.velocity_profile(9.46),0)

    def test_backward_ignores_forward_and_other_axes_and_x_clears(self):
        import torch
        with tempfile.NamedTemporaryFile(mode='w') as log, patch('deploy.deploy_real.go2_commission.assess_policy',return_value={'result':'zero_velocity_policy_trial_passed'}):
            log.write('{"mode":"policy","feedback":{"policy_entry_blend_seconds":0.5,"policy_pd_limit_nm":10.5}}');log.flush()
            test=PolicyCheck(self.cfg,log.name,policy=lambda obs:torch.zeros((1,12)),forward=True,forward_speed=.4,keyboard=True,keyboard_axis='backward')
        self.assertEqual(test.settle_tracking_tolerance, .25)
        test.started=1.;test.velocity_profile(9.)
        for key in ('w', 'a', 'd', 'q', 'e'):
            test.feed_key(key,9.)
            self.assertIsNone(test.keyboard_event)
        test.feed_key('s',9.)
        for i in range(1,21): command=test.velocity_profile(9.+i*.02)
        self.assertAlmostEqual(command[0],-.3,places=5)
        np.testing.assert_array_equal(command[1:],0)
        test.feed_key('x',9.41)
        np.testing.assert_array_equal(test.velocity_profile(9.42),0)
        test.feed_key('s',9.5);test.velocity_profile(9.52)
        np.testing.assert_array_equal(test.velocity_profile(10.11),0)

    def test_walk_turn_requires_explicit_forward_and_x_clears_both_axes(self):
        import torch
        with tempfile.NamedTemporaryFile(mode='w') as log, patch('deploy.deploy_real.go2_commission.assess_policy',return_value={'result':'zero_velocity_policy_trial_passed'}):
            log.write('{"mode":"policy","feedback":{"policy_entry_blend_seconds":0.5,"policy_pd_limit_nm":10.5}}');log.flush()
            test=PolicyCheck(self.cfg,log.name,policy=lambda obs:torch.zeros((1,12)),forward=True,forward_speed=.4,keyboard=True,keyboard_axis='yaw',yaw_speed=.5,turn_forward_speed=.3)
        test.started=1.;test.velocity_profile(9.);test.feed_key('e',9.)
        for i in range(1,26): v=test.velocity_profile(9.+i*.02)
        np.testing.assert_allclose(v,[.3,0.,-.5],atol=1e-6)
        test.feed_key('x',9.51)
        np.testing.assert_array_equal(test.velocity_profile(9.52),0)
        with patch('sys.stdin.isatty',return_value=True),redirect_stderr(StringIO()),self.assertRaises(SystemExit):
            parse_args(['eno1','--forward-check','ready','--keyboard','--forward-speed','0.4','--keyboard-axis','forward','--turn-forward-speed','0.3','--start-standing','--confirm-standing','--release-mode','--allow-body-movement','--allow-policy-movement'])

    def test_yaw_candidate_and_rejected_key_feedback(self):
        import torch
        with tempfile.NamedTemporaryFile(mode='w') as log, patch('deploy.deploy_real.go2_commission.assess_policy',return_value={'result':'zero_velocity_policy_trial_passed'}):
            log.write('{"mode":"policy","feedback":{"policy_entry_blend_seconds":0.5,"policy_pd_limit_nm":10.5}}');log.flush()
            test=PolicyCheck(self.cfg,log.name,policy=lambda obs:torch.zeros((1,12)),forward=True,forward_speed=.4,keyboard=True,keyboard_axis='yaw',yaw_speed=.5)
        test.started=1.;test.velocity_profile(9.);test.feed_key('q',9.)
        for i in range(1,26):test.velocity_profile(9.+i*.02)
        self.assertAlmostEqual(test.keyboard_velocity,.5)
        test.keyboard_used=4.;test.feed_key('e',9.52)
        self.assertFalse(test.keyboard_event['accepted'])
        self.assertEqual(test.keyboard_event['reason'],'budget_exhausted')
        test.feed_key('x',9.53)
        self.assertTrue(test.keyboard_event['accepted'])
        np.testing.assert_array_equal(test.velocity_profile(9.54),0)

    def test_walk_turn_real_cli_rejects_failed_yaw_candidates(self):
        base=['eno1','--forward-check','ready.jsonl','--keyboard','--forward-speed','0.4','--keyboard-axis','yaw','--turn-forward-speed','0.3','--start-standing','--confirm-standing','--release-mode','--allow-body-movement','--allow-policy-movement']
        with patch('sys.stdin.isatty',return_value=True):
            self.assertEqual(parse_args(base+['--yaw-speed','0.3']).yaw_speed,.3)
            for yaw in ('0.4','0.5'):
                with redirect_stderr(StringIO()),self.assertRaises(SystemExit):
                    parse_args(base+['--yaw-speed',yaw])

    def test_extended_keyboard_real_cli_duration(self):
        base=['eno1','--forward-check','ready.jsonl','--keyboard','--start-standing','--confirm-standing','--release-mode','--allow-body-movement','--allow-policy-movement']
        with patch('sys.stdin.isatty',return_value=True):
            self.assertEqual(parse_args(base+['--forward-speed','0.4']).duration,29)
            self.assertEqual(parse_args(base+['--forward-speed','0.3']).duration,23)
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(base+['--forward-speed','0.4','--duration','23'])

    def test_extended_keyboard_repeat_delay_and_stop(self):
        import torch
        with tempfile.NamedTemporaryFile(mode='w') as log, patch('deploy.deploy_real.go2_commission.assess_policy',return_value={'result':'zero_velocity_policy_trial_passed'}):
            log.write('{"mode":"policy","feedback":{"policy_entry_blend_seconds":0.5,"policy_pd_limit_nm":10.5}}');log.flush()
            test=PolicyCheck(self.cfg,log.name,policy=lambda obs:torch.zeros((1,12)),forward=True,forward_speed=.4,keyboard=True)
        test.started=1.
        self.assertEqual(test.duration,29.)
        test.velocity_profile(10.);test.feed_key('w',10.)
        for i in range(1,25): test.velocity_profile(10.+i*.02)
        self.assertAlmostEqual(test.keyboard_velocity,.4)
        test.feed_key('w',10.484);test.velocity_profile(10.5)
        self.assertAlmostEqual(test.keyboard_velocity,.4)
        test.feed_key('x',10.501)
        np.testing.assert_array_equal(test.velocity_profile(10.52),0)
        test.feed_key('w',10.6);test.velocity_profile(10.62)
        np.testing.assert_array_equal(test.velocity_profile(11.201),0)
        for i in range(250):
            now=11.3+i*.02;test.feed_key('w',now);test.velocity_profile(now)
        self.assertTrue(4<=test.keyboard_used<=4.03)
        self.assertEqual(test.keyboard_velocity,0)
        np.testing.assert_array_equal(test.velocity_profile(21.),0)

    def test_prone_passive_calves_do_not_change_policy_limits(self):
        before = copy.deepcopy(self.cfg.safety)
        check_prone(self.cfg, self.prone(), 1.0)
        self.assertEqual(self.cfg.safety, before)
        standing = self.prone()
        standing.joint_pos_sdk[:] = self.cfg.default_joint_pos[self.cfg.sdk_to_policy]
        with self.assertRaisesRegex(ValueError, '禁止站立'):
            check_prone(self.cfg, standing, 1.0)

    def test_one_hip_only_and_bounded_torque(self):
        test = ProneTest(self.cfg, 'FL_hip_joint', .005, 2, True)
        test.arm(1, self.prone())
        command = test.output(2, self.prone(2))
        self.assertEqual(command.mode, Go2Mode.MOTOR_TEST)
        self.assertEqual(np.flatnonzero(command.kp_sdk).tolist(), [3])
        self.assertAlmostEqual(command.target_pos_sdk[3] - self.prone().joint_pos_sdk[3], .005, places=6)
        np.testing.assert_array_equal(np.delete(command.target_pos_sdk, 3), np.delete(test.initial, 3))
        np.testing.assert_array_equal(command.kd_sdk, .5)
        test.stop(2.1, '第二终端请求停止')
        np.testing.assert_array_equal(test.output(2.2, self.prone(2.2)).kp_sdk, 0)
        with self.assertRaises(ValueError):
            test.arm(2.2, self.prone(2.2))

    def test_damping_mode_never_has_positive_kp_and_expires(self):
        test = ProneTest(self.cfg, 'FL_hip_joint', .005, 1, False)
        test.arm(1, self.prone())
        np.testing.assert_array_equal(test.output(1.5, self.prone(1.5)).kp_sdk, 0)
        command = test.output(2, self.prone(2))
        self.assertEqual(command.mode, Go2Mode.DAMPING)
        np.testing.assert_array_equal(command.kd_sdk, 3)

    def test_state_faults_latch_damping(self):
        for kind in ('stale', 'displacement', 'velocity', 'nan', 'pose'):
            with self.subTest(kind=kind):
                test = ProneTest(self.cfg, 'FL_hip_joint', .005, 2, True)
                test.arm(1, self.prone())
                state = self.prone(1.1)
                if kind == 'stale':
                    state.timestamp = 0
                elif kind == 'displacement':
                    state.joint_pos_sdk[0] += .05
                elif kind == 'velocity':
                    state.joint_vel_sdk[0] = 3
                elif kind == 'nan':
                    state.joint_pos_sdk[0] = np.nan
                elif kind == 'pose':
                    state.quaternion_wxyz[:] = [0, 1, 0, 0]
                command = test.output(1.1, state)
                self.assertTrue(command.reason)
                self.assertEqual(command.mode, Go2Mode.DAMPING)
                np.testing.assert_array_equal(command.kp_sdk, 0)
                self.assertEqual(test.output(1.2, self.prone(1.2)).reason, command.reason)

    def test_unsafe_cli_and_parameters_rejected(self):
        for argv in (['eno1', '--motor-test'], ['lo', '--motor-test', '--confirm-prone', '--release-mode'],
                     ['eno1', '--motor-test', '--confirm-prone'], ['eno1', '--damping-test', '--motor-test'],
                     ['eno1', '--simulation-auto']):
            with self.subTest(argv=argv), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)
        for joint, amplitude, duration in (('FL_calf_joint', .005, 2), ('FL_hip_joint', .011, 2),
                                          ('FL_hip_joint', float('nan'), 2), ('FL_hip_joint', .005, 4)):
            with self.assertRaises(ValueError):
                ProneTest(self.cfg, joint, amplitude, duration, True)

    def test_restore_requires_factory_stand_acknowledgement(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(['eno1', '--restore-normal', '--confirm-prone'])
        args = parse_args(['eno1', '--restore-normal', '--confirm-prone', '--allow-factory-stand'])
        self.assertTrue(args.restore_normal and args.allow_factory_stand)

    def test_response_profile_bounds_and_feedback_guards(self):
        def state(now):
            s = self.prone(now)
            s.feedback = {'joint_tau_est_sdk': [0.] * 12}
            return s
        test = ProneTest(self.cfg, 'FL_hip_joint', .005, 3, True, response=True)
        test.arm(1, state(1))
        for now, expected in ((1.2, 0), (1.8, .005), (2.4, 0), (3., -.005), (3.8, 0)):
            cmd = test.output(now, state(now))
            self.assertAlmostEqual(cmd.target_pos_sdk[3] - test.initial[3], expected, places=6)
            self.assertEqual(np.flatnonzero(cmd.kp_sdk).tolist(), [3])
            self.assertEqual(cmd.kp_sdk[3], 20)
        for kind in ('tau', 'velocity', 'position', 'estimated_torque', 'missing_feedback'):
            with self.subTest(kind=kind):
                test = ProneTest(self.cfg, 'FL_hip_joint', .005, 3, True, response=True)
                test.arm(1, state(1))
                s = state(1.8)
                if kind == 'tau':
                    s.feedback['joint_tau_est_sdk'][3] = .6
                elif kind == 'velocity':
                    s.joint_vel_sdk[3] = .6
                elif kind == 'position':
                    s.joint_pos_sdk[3] += .011
                elif kind == 'estimated_torque':
                    s.joint_pos_sdk[3] -= .008
                else:
                    s.feedback = None
                self.assertEqual(test.output(1.8, s).mode, Go2Mode.DAMPING)
        for amplitude, duration in ((.01, 3), (-.005, 3), (.005, 2)):
            with self.assertRaises(ValueError):
                ProneTest(self.cfg, 'FL_hip_joint', amplitude, duration, True, response=True)

    def test_joint_check_all_twelve_and_calf_inward_transition(self):
        limits = copy.deepcopy(self.cfg.safety)
        for name in self.cfg.data['policy']['joint_names']:
            with self.subTest(joint=name):
                def state(now):
                    s = self.prone(now)
                    s.feedback = {'joint_tau_est_sdk': [0.] * 12}
                    return s
                test = ProneTest(self.cfg, name, .02, 3, True, joint_check=True)
                test.arm(1, state(1))
                for now, sign in ((1.8, 1), (3., 1 if test.recover_range else -1)):
                    cmd = test.output(now, state(now))
                    self.assertEqual(cmd.mode, Go2Mode.MOTOR_TEST)
                    self.assertEqual(np.flatnonzero(cmd.kp_sdk).tolist(), [test.joint])
                    self.assertEqual(cmd.kp_sdk[test.joint], 40)
                    self.assertAlmostEqual(cmd.target_pos_sdk[test.joint] - test.initial[test.joint], sign * .02, places=6)
                    np.testing.assert_array_equal(np.delete(cmd.target_pos_sdk, test.joint), np.delete(test.initial, test.joint))
                self.assertEqual(test.recover_range, name.endswith('_calf_joint'))
                if test.recover_range:
                    s = state(3.1)
                    s.joint_pos_sdk[test.joint] -= .0031
                    self.assertEqual(test.output(3.1, s).mode, Go2Mode.DAMPING)
        self.assertEqual(self.cfg.safety, limits)

    def test_joint_check_feedback_guards_and_fixed_parameters(self):
        for kind in ('tau', 'velocity', 'position', 'estimated_torque', 'missing_feedback'):
            with self.subTest(kind=kind):
                s = self.prone()
                s.feedback = {'joint_tau_est_sdk': [0.] * 12}
                test = ProneTest(self.cfg, 'FL_hip_joint', .02, 3, True, joint_check=True)
                test.arm(1, s)
                s.timestamp = 1.8
                if kind == 'tau':
                    s.feedback['joint_tau_est_sdk'][3] = 2.01
                elif kind == 'velocity':
                    s.joint_vel_sdk[0] = .51
                elif kind == 'position':
                    s.joint_pos_sdk[0] += .031
                elif kind == 'estimated_torque':
                    s.joint_pos_sdk[3] -= .006
                else:
                    s.feedback = None
                self.assertEqual(test.output(1.8, s).mode, Go2Mode.DAMPING)
                self.assertTrue(test.reason)
        for amplitude, duration in ((.03, 3), (-.02, 3), (.02, 2)):
            with self.assertRaises(ValueError):
                ProneTest(self.cfg, 'FL_hip_joint', amplitude, duration, True, joint_check=True)

    def test_joint_check_analyzer_uses_selected_calf_and_repeat_direction(self):
        s = self.prone()
        s.feedback = {'joint_tau_est_sdk': [0.] * 12}
        test = ProneTest(self.cfg, 'RR_calf_joint', .02, 3, True, joint_check=True)
        test.arm(1, s)
        rows = []
        for tick in range(1500):
            now = 1 + tick * .002
            phase, offset = test.phase(now)
            q, target, tau = s.joint_pos_sdk.copy(), s.joint_pos_sdk.copy(), [0.] * 12
            q[test.joint] += offset
            target[test.joint] += offset
            tau[test.joint] = 40 * offset
            rows.append({'time': now, 'mode': 'motor_test', 'reason': '', 'joint_pos_sdk': q.tolist(),
                         'target_pos_sdk': target.tolist(), 'feedback': {'state_tick': tick,
                         'test_joint_sdk': test.joint, 'range_transition': True, 'test_phase': phase,
                         'joint_tau_est_sdk': tau, 'observed_lowcmd': {'crc_valid': True}}})
        result = assess_response(rows, test.joint)
        self.assertTrue(result['force_response'] and result['encoder_response'])
        self.assertFalse(result['physical_joint_direction_confirmed'])

    def test_response_analysis_separates_force_encoder_and_no_response(self):
        for kind in ('none', 'force', 'encoder', 'both'):
            with self.subTest(kind=kind):
                rows = []
                test = ProneTest(self.cfg, 'FL_hip_joint', .005, 3, True, response=True)
                s = self.prone()
                s.feedback = {'joint_tau_est_sdk': [0.] * 12}
                test.arm(1, s)
                for tick in range(1500):
                    now = 1 + tick * .002
                    phase, delta = test.phase(now)
                    q = s.joint_pos_sdk.copy()
                    tau = [0.] * 12
                    if kind in ('encoder', 'both'):
                        q[3] += delta
                    if kind in ('force', 'both'):
                        tau[3] = 20 * delta
                    target = s.joint_pos_sdk.copy()
                    target[3] += delta
                    rows.append({'time': now, 'mode': 'motor_test', 'reason': '', 'joint_pos_sdk': q.tolist(),
                                 'target_pos_sdk': target.tolist(), 'feedback': {'state_tick': tick,
                                 'test_phase': phase, 'joint_tau_est_sdk': tau,
                                 'observed_lowcmd': {'crc_valid': True}}})
                result = assess_response(rows)
                self.assertEqual(result['force_response'], kind in ('force', 'both'))
                self.assertEqual(result['encoder_response'], kind in ('encoder', 'both'))
                self.assertFalse(result['physical_joint_direction_confirmed'])
                # 静止记录、重复tick、错误CRC不能当成完整响应。
                self.assertEqual(assess_response(rows[:100])['result'], 'inconclusive')
                broken = copy.deepcopy(rows)
                for row in broken:
                    row['feedback']['state_tick'] = 0
                self.assertEqual(assess_response(broken)['result'], 'inconclusive')
                rows[0]['reason'] = '第二终端请求停止'
                self.assertEqual(assess_response(rows)['result'], 'aborted')

    def test_mode_release_fails_closed(self):
        client = Mock()
        client.CheckMode.return_value = (1, None)
        with self.assertRaises(RuntimeError):
            release_mode(client)
        client.ReleaseMode.assert_not_called()
        client.CheckMode.return_value = (0, {'name': 'ai'})
        with self.assertRaises(RuntimeError):
            release_mode(client)
        client.ReleaseMode.assert_not_called()
        client.CheckMode.return_value = (0, {'name': 'normal'})
        client.ReleaseMode.return_value = (5, None)
        with self.assertRaisesRegex(RuntimeError, '退出原厂'):
            release_mode(client)
        client.reset_mock()
        client.CheckMode.side_effect = [(0, {'name': 'normal'}), (0, {'name': ''})]
        client.ReleaseMode.return_value = (0, None)
        self.assertEqual(release_mode(client), 'normal')
        client.ReleaseMode.assert_called_once()

    def test_stand_check_full_profile_never_runs_policy_and_lowers(self):
        test = StandCheck(self.cfg)
        s = self.prone()
        s.feedback = {'joint_tau_est_sdk': [0.] * 12, 'state_tick': 0}
        test.arm(1, s)
        rows = []
        for tick in range(3401):
            now = 1 + tick * .005
            target, kp, kd = test.profile(now)
            s = Go2State(now, np.array([1., 0, 0, 0]), np.zeros(3), target.copy(), np.zeros(12))
            s.feedback = {'joint_tau_est_sdk': [0.] * 12, 'state_tick': tick,
                          'test_phase': test.phase(now)[0], 'test_elapsed': now - 1,
                          'observed_lowcmd': {'crc_valid': True}}
            cmd = test.output(now, s)
            self.assertEqual(cmd.mode, Go2Mode.DAMPING if tick == 3400 else Go2Mode.FIX_STAND)
            if 9 <= now - 1 < 10:
                np.testing.assert_allclose(kp, self.cfg.data['fsm']['stand_kp'])
                np.testing.assert_allclose(kd, self.cfg.data['fsm']['stand_kd'])
            rows.append({'time': now, 'mode': cmd.mode.value, 'reason': cmd.reason,
                         'kp_sdk': cmd.kp_sdk.tolist(), 'kd_sdk': cmd.kd_sdk.tolist(),
                         'joint_pos_sdk': s.joint_pos_sdk.tolist(), 'joint_vel_sdk': s.joint_vel_sdk.tolist(),
                         'quaternion_wxyz': s.quaternion_wxyz.tolist(), 'feedback': s.feedback})
        self.assertEqual(test.reason, '测试时长结束')
        self.assertEqual(assess_stand(rows, self.cfg)['result'], 'stand_hold_lower_passed')
        self.assertEqual(assess_stand(rows[:100], self.cfg)['result'], 'inconclusive')
        rows[-1]['reason'] = '第二终端请求停止'
        self.assertEqual(assess_stand(rows, self.cfg)['result'], 'aborted')

    def test_stand_check_faults_latch_and_require_real_progress(self):
        for kind in ('stale', 'loop', 'tilt', 'velocity', 'torque', 'tracking', 'range_entry', 'missing_feedback'):
            with self.subTest(kind=kind):
                test = StandCheck(self.cfg)
                s = self.prone()
                s.feedback = {'joint_tau_est_sdk': [0.] * 12, 'state_tick': 0}
                test.arm(1, s)
                self.assertEqual(test.output(1, s).mode, Go2Mode.FIX_STAND)
                now = 1.01
                s.timestamp = now
                if kind == 'stale':
                    s.timestamp = 0
                elif kind == 'loop':
                    now = s.timestamp = 1.06
                elif kind == 'tilt':
                    s.quaternion_wxyz[:] = [0, 1, 0, 0]
                elif kind == 'velocity':
                    s.joint_vel_sdk[0] = 2.01
                elif kind == 'torque':
                    s.feedback['joint_tau_est_sdk'][0] = StandCheck.torque_limit + .01
                elif kind == 'tracking':
                    s.joint_pos_sdk[0] += .251
                elif kind == 'range_entry':
                    s.joint_pos_sdk[2] -= .0031
                else:
                    s.feedback = None
                cmd = test.output(now, s)
                self.assertEqual(cmd.mode, Go2Mode.DAMPING)
                np.testing.assert_array_equal(cmd.kp_sdk, 0)
                self.assertEqual(test.output(now + .01, self.prone(now + .01)).reason, cmd.reason)
        # 无新tick的假静态帧不能满足站立保持。
        test = StandCheck(self.cfg)
        s = self.prone()
        s.feedback = {'joint_tau_est_sdk': [0.] * 12, 'state_tick': 0}
        test.arm(1, s)
        for tick in range(2001):
            now = 1 + tick * .005
            s.timestamp = now
            s.joint_pos_sdk = test.profile(now)[0]
            cmd = test.output(now, s)
        self.assertEqual(cmd.mode, Go2Mode.DAMPING)
        self.assertIn('保持', cmd.reason)

    def test_stand_cli_requires_explicit_body_movement_and_fixed_duration(self):
        for argv in (['eno1', '--stand-check', '--confirm-prone', '--release-mode'],
                     ['eno1', '--stand-check', '--confirm-prone', '--release-mode', '--allow-body-movement', '--duration', '30']):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)
        with patch('sys.stdin.isatty', return_value=True):
            args = parse_args(['eno1', '--stand-check', '--confirm-prone', '--release-mode', '--allow-body-movement'])
        self.assertEqual(args.duration, 17)
        with patch('sys.stdin.isatty', return_value=True):
            args = parse_args(['eno1', '--stand-check', '--start-standing', '--confirm-standing', '--release-mode', '--allow-body-movement'])
        self.assertEqual(args.duration, 14)

    def test_standing_start_preserves_initial_pose_and_has_no_prone_transition(self):
        test = StandCheck(self.cfg, start_standing=True)
        s = self.prone()
        s.feedback = {'joint_tau_est_sdk': [0.] * 12, 'state_tick': 0}
        with self.assertRaisesRegex(ValueError, '站立接管范围'):
            test.arm(1, s)
        s.joint_pos_sdk = np.tile([0., .8, -1.5], 4)
        test.arm(1, s)
        first = test.output(1, s)
        np.testing.assert_allclose(first.target_pos_sdk, s.joint_pos_sdk)
        np.testing.assert_allclose(first.kp_sdk, 60)
        for tick in range(1, 2801):
            now = 1 + tick * .005
            s.timestamp = now
            s.joint_pos_sdk = test.profile(now)[0]
            s.feedback['state_tick'] = tick
            cmd = test.output(now, s)
            self.assertNotEqual(test.phase(now)[0], 'range_entry')
            self.assertEqual(cmd.mode, Go2Mode.DAMPING if tick == 2800 else Go2Mode.FIX_STAND)
        self.assertEqual(test.reason, '测试时长结束')

    def test_handoff_foreign_command_detection(self):
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from collections import deque
        transport = VerifiedTransport.__new__(VerifiedTransport)
        transport._lock = threading.Lock()
        transport._observed_lowcmd_count = 0
        transport._crc = SimpleNamespace(Crc=lambda _: 123)
        transport.guard_command_source = True
        transport.foreign_command = False
        m = unitree_go_msg_dds__LowCmd_()
        m.crc = 123
        q, kp, kd = [0.] * 12, [60.] * 12, [5.] * 12
        for i in range(12):
            m.motor_cmd[i].q, m.motor_cmd[i].kp, m.motor_cmd[i].kd = q[i], kp[i], kd[i]
        transport._sent_signatures = deque([transport.signature(q, kp, kd)])
        transport._on_command(m)
        self.assertFalse(transport.foreign_command)
        m.motor_cmd[0].q = .1
        transport._on_command(m)
        self.assertTrue(transport.foreign_command)

    def test_startup_lease_requires_live_parent_and_honors_stop(self):
        import json
        for kind in ('missing', 'stale', 'fresh', 'stop'):
            with self.subTest(kind=kind):
                test = StandCheck(self.cfg, start_standing=True)
                server = Mock()
                payloads = []
                if kind != 'missing':
                    payloads.append((json.dumps({'type': 'startup_heartbeat', 'time': time.monotonic() - (1 if kind == 'stale' else 0)}).encode(), ''))
                if kind == 'stop':
                    payloads.append((b'{"type":"stop"}', 'stop-client'))
                server.recvfrom.side_effect = payloads + [BlockingIOError()]
                startup_lease(server, test)
                self.assertEqual(bool(test.reason), kind != 'fresh')
                if kind == 'stop':
                    reply = json.loads(server.sendto.call_args.args[0])
                    self.assertTrue(reply['stopped'])
                    self.assertFalse(reply['dds_submitted'])

    def test_control_gc_restored_after_error_and_respects_disabled_caller(self):
        import gc
        original = gc.isenabled()
        try:
            gc.enable()
            with self.assertRaises(RuntimeError):
                with bounded_control_gc():
                    self.assertFalse(gc.isenabled())
                    raise RuntimeError('模拟异常')
            self.assertTrue(gc.isenabled())
            gc.disable()
            with bounded_control_gc():
                self.assertFalse(gc.isenabled())
            self.assertFalse(gc.isenabled())
        finally:
            gc.enable() if original else gc.disable()

    def test_bad_crc_does_not_refresh_state(self):
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
        transport = VerifiedTransport.__new__(VerifiedTransport)
        transport._crc = SimpleNamespace(Crc=lambda _: 123)
        transport._lock = threading.Lock()
        transport._state = None
        transport._on_state(SimpleNamespace(crc=0))
        self.assertIsNone(transport.snapshot())
        message = unitree_go_msg_dds__LowState_()
        message.crc = 123
        message.tick = 42
        message.motor_state[3].tau_est = .25
        message.foot_force = [11, 22, 33, 44]
        message.foot_force_est = [1, 2, 3, 4]
        message.imu_state.accelerometer = [0., 0., 9.81]
        transport._on_state(message)
        state = transport.snapshot()
        self.assertEqual(state.feedback['state_tick'], 42)
        self.assertEqual(state.feedback['joint_tau_est_sdk'][3], .25)
        self.assertEqual(state.feedback['foot_force_raw'], [11, 22, 33, 44])
        self.assertEqual(state.feedback['foot_force_est_raw'], [1, 2, 3, 4])
        np.testing.assert_allclose(state.feedback['imu_accelerometer_raw'], [0., 0., 9.81])
        message.foot_force[0] = 99
        self.assertEqual(state.feedback['foot_force_raw'][0], 11)
        message.crc = 0
        message.tick = 99
        transport._on_state(message)
        self.assertEqual(transport.snapshot().feedback['state_tick'], 42)

    def test_server_stop_and_missing_heartbeat_publish_damping(self):
        # 使用真实Unix停止接口和计时循环，DDS与原厂RPC均替换为测试对象。
        for scenario in ('stop', 'heartbeat_loss'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory) / 'stop.sock')
                sent, errors = [], []
                outer = self

                class Transport:
                    def __init__(self, config, publish):
                        self.publish = publish

                    def snapshot(self):
                        return outer.prone(time.monotonic())

                    def send(self, command):
                        sent.append((time.monotonic(), command))

                cfg = copy.deepcopy(self.cfg)
                cfg.data['recording']['enabled'] = False
                args = SimpleNamespace(network='eno1', test_joint='FL_hip_joint', test_amplitude=.005,
                                       duration=2, motor_test=True)

                def worker():
                    try:
                        serve(args, cfg)
                    except Exception as error:
                        errors.append(error)

                with patch('deploy.deploy_real.go2_commission.default_socket', return_value=path), \
                     patch('deploy.deploy_real.go2_commission.VerifiedTransport', Transport), \
                     patch('deploy.deploy_real.go2_commission.release_mode', return_value='normal'), \
                     patch('deploy.deploy_real.go2_commission.mode_client'), \
                     patch('unitree_sdk2py.core.channel.ChannelFactoryInitialize'), \
                     patch('deploy.deploy_real.go2_commission.signal.signal'):
                    thread = threading.Thread(target=worker)
                    thread.start()
                    deadline = time.monotonic() + 4
                    while time.monotonic() < deadline:
                        try:
                            if request(path, {'type': 'status'}, timeout=.05)['ready']:
                                break
                        except OSError:
                            pass
                        time.sleep(.01)
                    else:
                        thread.join(timeout=1)
                        self.fail(f'验收进程未就绪: {errors!r}')
                    import socket
                    import json
                    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
                        trigger = time.monotonic()
                        sender.sendto(json.dumps({'type': 'arm', 'time': trigger}).encode(), path)
                    deadline = time.monotonic() + .2
                    while not sent and time.monotonic() < deadline:
                        time.sleep(.005)
                    self.assertTrue(sent)
                    if scenario == 'stop':
                        trigger = time.monotonic()
                        reply = request(path, {'type': 'stop'})
                        self.assertTrue(reply['stopped'])
                        self.assertTrue(any(c.mode is Go2Mode.DAMPING for _, c in sent))
                    thread.join(timeout=4)
                    self.assertFalse(thread.is_alive())
                    damp = [(t, c) for t, c in sent if c.mode is Go2Mode.DAMPING]
                    self.assertTrue(damp)
                    self.assertLess(damp[0][0] - trigger, .2)
                    self.assertGreaterEqual(damp[-1][0] - damp[0][0], .98)
                    first = next(i for i, (_, c) in enumerate(sent) if c.mode is Go2Mode.DAMPING)
                    self.assertTrue(all(np.max(c.kp_sdk) == 0 for _, c in sent[first:]))
                    self.assertEqual(bool(errors), scenario == 'heartbeat_loss')
                    self.assertFalse(Path(path).exists())


@unittest.skipUnless(os.environ.get('GO2_DDS_TESTS') == '1', '设置GO2_DDS_TESTS=1执行本机DDS验收入口测试')
class Go2CommissionDDSTest(unittest.TestCase):
    def test_actual_sdk_crc_motor_and_stop_on_loopback(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        import socket
        import json

        # 直接测试serve，替换原厂RPC；DDS真正使用lo及非零domain，绝不使用eno1/domain0。
        ChannelFactoryInitialize(180 + os.getpid() % 30, 'lo')
        publisher = ChannelPublisher('rt/lowstate', LowState_)
        publisher.Init()
        received = []
        start_standing = os.environ.get('GO2_START_STANDING_TEST') == '1'
        stand_check = os.environ.get('GO2_STAND_CHECK_TEST') == '1' or start_standing
        crc = CRC()
        subscriber = ChannelSubscriber('rt/lowcmd', LowCmd_)
        subscriber.Init(lambda message: received.append((time.monotonic(), message)), 100)
        done = threading.Event()
        state = unitree_go_msg_dds__LowState_()
        state.imu_state.quaternion = [1., 0., 0., 0.]
        initial = Go2CommissionTest().prone().joint_pos_sdk
        if start_standing:
            initial = np.tile([0., .8, -1.5], 4)
        for i in range(12):
            state.motor_state[i].q = float(initial[i])

        release_at = [None]

        def states():
            while not done.is_set():
                if start_standing:
                    # 原厂释放后短时速度升高；起点验证应发生在释放前。
                    moving = release_at[0] is not None and time.monotonic() - release_at[0] < .05
                    for i in range(12):
                        state.motor_state[i].dq = .4 if moving and i % 3 == 2 else 0.
                if stand_check and received:
                    message = received[-1][1]
                    if all(x.kp > 0 for x in message.motor_cmd[:12]):
                        for i in range(12):
                            state.motor_state[i].q = message.motor_cmd[i].q
                state.tick += 1
                state.crc = crc.Crc(state)
                publisher.Write(state)
                time.sleep(.002)

        producer = threading.Thread(target=states)
        producer.start()
        errors = []
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'stop.sock')
            cfg = Go2RealConfig.load(ROOT / 'deploy/deploy_real/configs/go2.yaml', str(ROOT))
            cfg.data['recording']['enabled'] = True
            cfg.data['recording']['directory'] = str(Path(directory) / 'records')
            joint_check = os.environ.get('GO2_JOINT_CHECK_TEST') == '1'
            selected = 8 if joint_check else 3
            args = SimpleNamespace(network='lo', test_joint='RR_calf_joint' if joint_check else 'FL_hip_joint',
                                   test_amplitude=.02 if joint_check else .005,
                                   duration=(14 if start_standing else 17) if stand_check else 3, motor_test=not stand_check,
                                   response_test=not joint_check and not stand_check, joint_check=joint_check and not stand_check,
                                   stand_check=stand_check, start_standing=start_standing)

            def worker():
                try:
                    serve(args, cfg)
                except Exception as error:
                    errors.append(error)

            def release_with_motion(_client):
                release_at[0] = time.monotonic()
                if start_standing:
                    time.sleep(.012)
                return 'normal'

            thread = threading.Thread(target=worker)
            try:
                with patch('deploy.deploy_real.go2_commission.default_socket', return_value=path), \
                     patch('deploy.deploy_real.go2_commission.mode_client'), \
                     patch('deploy.deploy_real.go2_commission.release_mode', side_effect=release_with_motion), \
                     patch('deploy.deploy_real.go2_commission.read_mode', return_value='normal'), \
                     patch('unitree_sdk2py.core.channel.ChannelFactoryInitialize'), \
                     patch('deploy.deploy_real.go2_commission.signal.signal'):
                    thread.start()
                    deadline = time.monotonic() + 6
                    while time.monotonic() < deadline:
                        try:
                            if start_standing:
                                with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as startup:
                                    startup.sendto(json.dumps({'type': 'startup_heartbeat', 'time': time.monotonic()}).encode(), path)
                            if request(path, {'type': 'status'}, timeout=.05)['ready']:
                                break
                        except OSError:
                            pass
                        time.sleep(.01)
                    else:
                        self.fail(f'DDS验收进程未就绪: {errors!r}')
                    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
                        sender.sendto(json.dumps({'type': 'arm', 'time': time.monotonic()}).encode(), path)
                        until = time.monotonic() + ((13.6 if start_standing else 16.6) if stand_check else 2.6)
                        while time.monotonic() < until:
                            sender.sendto(json.dumps({'type': 'heartbeat', 'time': time.monotonic()}).encode(), path)
                            time.sleep(.02)
                    trigger = time.monotonic()
                    self.assertTrue(request(path, {'type': 'stop'})['stopped'])
                    thread.join(timeout=4)
                    self.assertFalse(thread.is_alive())
                self.assertFalse(errors)
                logs = list((Path(directory) / 'records/commission').glob('*.jsonl'))
                self.assertEqual(len(logs), 1)
                rows = [json.loads(line) for line in logs[0].read_text().splitlines()]
                self.assertTrue(all('joint_tau_est_sdk' in row['feedback'] for row in rows))
                echoes = [row['feedback']['observed_lowcmd'] for row in rows if row['feedback']['observed_lowcmd']]
                self.assertTrue(echoes)
                self.assertTrue(all(echo['crc_valid'] for echo in echoes))
                active = [m for _, m in received if any(x.kp > 0 for x in m.motor_cmd[:12])]
                self.assertTrue(active)
                if stand_check:
                    self.assertTrue(all(all(40 <= x.kp <= 60 for x in m.motor_cmd[:12]) for m in active))
                    self.assertTrue(any(all(abs(x.kp - 40) < 1e-4 for x in m.motor_cmd[:12]) for m in active))
                    self.assertEqual({r['feedback']['test_phase'] for r in rows}, {'rise', 'hold', 'lower', 'settle'} | (set() if start_standing else {'range_entry'}))
                    self.assertFalse(any(r['mode'] == 'policy' for r in rows))
                    np.testing.assert_allclose([m.q for m in active[-1].motor_cmd[:12]], StandCheck(cfg, start_standing).lowered, atol=.001)
                else:
                    self.assertTrue(all([i for i, x in enumerate(m.motor_cmd[:12]) if x.kp > 0] == [selected] for m in active))
                    self.assertTrue(all(m.motor_cmd[selected].kp == (40 if joint_check else 20) for m in active))
                    deltas = [m.motor_cmd[selected].q - initial[selected] for m in active]
                    self.assertGreater(max(deltas), .0199 if joint_check else .0049)
                if joint_check and not stand_check:
                    self.assertGreaterEqual(min(deltas), -1e-6)
                    self.assertTrue(any(row['feedback']['test_phase'] == 'positive_repeat' for row in rows))
                elif not stand_check:
                    self.assertLess(min(deltas), -.0049)
                self.assertTrue(all(m.crc == crc.Crc(m) for _, m in received))
                damping = [(t, m) for t, m in received if all(x.kp == 0 and x.kd == 3 for x in m.motor_cmd[:12])]
                self.assertTrue(damping)
                self.assertLess(damping[0][0] - trigger, .2)
                self.assertGreaterEqual(damping[-1][0] - damping[0][0], .98)
                first = next(i for i, (_, m) in enumerate(received) if all(x.kp == 0 for x in m.motor_cmd[:12]))
                self.assertTrue(all(all(x.kp == 0 for x in m.motor_cmd[:12]) for _, m in received[first:]))
            finally:
                if thread.is_alive():
                    try:
                        request(path, {'type': 'stop'}, timeout=.2)
                    except OSError:
                        pass
                    thread.join(timeout=4)
                done.set()
                producer.join(timeout=2)
                subscriber.Close()
                publisher.Close()


if __name__ == '__main__':
    unittest.main()
