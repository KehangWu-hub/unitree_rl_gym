"""Joint-only stall detection and bounded, cancellable policy reentry."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from deploy.common.go2_gait_restart import GaitRestart
from deploy.deploy_real import go2_commission as commission
from deploy.deploy_real import go2_keyboard as entry
from deploy.deploy_real.deploy_real_go2 import Go2Command, Go2Mode, Go2RealConfig, Go2State

ROOT=Path(__file__).resolve().parents[1]


class GaitRestartTest(unittest.TestCase):
    def test_stalled_log_like_motion_triggers_but_walking_and_body_rotation_do_not(self):
        for scenario in ['static','walking','rotating','wrong_direction']:
            restart=GaitRestart();result=None
            for n in range(60):
                t=1+n*.02
                q=np.full(12,.0004*np.sin(n))
                dq=np.full(12,.036)
                gyro=np.zeros(3)
                target=[0,.3,0]
                if scenario=='walking':q[:]=.04*np.sin(2*np.pi*t/.6);dq[:]=.4
                if scenario=='rotating':gyro[2]=.2
                if scenario=='wrong_direction':target=[0,-.3,0]
                result=restart.observe(t,target,q,dq,gyro)
                if result:break
            self.assertEqual(result,'restart' if scenario=='static' else None)

    def test_repeated_sensor_frame_cannot_trigger_recovery(self):
        restart=GaitRestart()
        for n in range(100):
            event=restart.observe(1+n*.02,[0,.3,0],np.zeros(12),np.zeros(12),np.zeros(3),sensor_tick=1)
            self.assertIsNone(event)
        self.assertEqual(restart.count,0)

    def test_one_attempt_then_block_until_explicit_reset(self):
        restart=GaitRestart()
        c=Go2Command(Go2Mode.POLICY,np.zeros(12),np.ones(12)*20,np.ones(12)*.5)
        restart.begin(1.,c)
        restart.phase='monitor'
        result=None
        for n in range(60):
            result=restart.observe(5+n*.02,[0,.3,0],np.zeros(12),np.zeros(12),np.zeros(3))
        self.assertEqual(result,'blocked')
        self.assertTrue(restart.blocked)
        self.assertEqual(restart.count,1)
        restart.reset_request()
        self.assertFalse(restart.blocked)
        self.assertFalse(restart.attempted)

    def test_stability_is_required_before_reentry_and_timeout_is_latched_by_caller(self):
        restart=GaitRestart();q=np.zeros(12)
        c=Go2Command(Go2Mode.POLICY,q.copy(),np.ones(12)*20,np.ones(12)*.5)
        restart.begin(1.,c)
        args=(q,np.zeros(12),q,np.ones(12)*40,np.ones(12),0.)
        self.assertFalse(restart.stand_command(2.,*args)[3])
        self.assertFalse(restart.stand_command(3.,*args)[3])
        self.assertTrue(restart.stand_command(3.81,*args)[3])
        restart=GaitRestart();restart.begin(1.,c)
        with self.assertRaisesRegex(ValueError,'稳定站姿'):
            restart.stand_command(6.1,q,np.ones(12),q,np.ones(12)*40,np.ones(12),0.)

    def trial(self):
        config=Go2RealConfig.load(ROOT/'deploy/deploy_real/configs/go2.yaml',str(ROOT))
        class Actor:
            def __init__(self):self.observations=[]
            def __call__(self,obs):
                self.observations.append(obs.numpy().copy())
                return torch.ones((1,12))*.2
        actor=Actor()
        with tempfile.NamedTemporaryFile(mode='w') as f, patch.object(commission,'assess_policy',return_value={'result':'zero_velocity_policy_trial_passed'}):
            f.write(json.dumps({'mode':'policy','feedback':{'policy_entry_blend_seconds':.5,'policy_pd_limit_nm':14.,'policy_feedback_limit_nm':16.}}));f.flush()
            test=commission.PolicyCheck(config,f.name,policy=actor,forward=True,forward_speed=.4,
                                        keyboard=True,keyboard_axis='all',turn_forward_speed=.3,continuous=True)
        test.gait_restart=GaitRestart()
        q=config.default_joint_pos[config.sdk_to_policy]
        def state(now,tick):
            s=Go2State(now,np.array([1.,0,0,0]),np.zeros(3),q.copy(),np.zeros(12))
            s.feedback={'joint_tau_est_sdk':[0.]*12,'state_tick':tick}
            return s
        test.arm(1.,state(1.,0))
        return test,actor,state

    def test_real_policy_does_not_reset_history_until_stand_is_stable(self):
        test,actor,state=self.trial();reentry_inference=None
        for tick in range(10001):
            now=1+tick*.002
            if tick%10==0 and now>=9.:test.feed_key('a',now)
            before=test.policy_reentry_at
            command=test.output(now,state(now,tick))
            self.assertNotEqual(command.mode,Go2Mode.DAMPING,command.reason)
            self.assertLessEqual(float(np.max(abs(command.kp_sdk*(command.target_pos_sdk-state(now,tick).joint_pos_sdk)))),14.00001)
            if before is None and test.policy_reentry_at is not None:
                reentry_inference=actor.observations[-1]
                self.assertGreaterEqual(now-test.gait_restart.started,2.8-1e-6)
            if test.gait_restart.blocked:break
        self.assertEqual(test.gait_restart.count,1)
        self.assertTrue(test.gait_restart.blocked)
        self.assertIsNotNone(reentry_inference)
        np.testing.assert_array_equal(reentry_inference[0,33:45],0.)
        np.testing.assert_array_equal(test.controller.command,0.)
        test.stop(now,'Z测试')
        self.assertEqual(test.output(now+.002,state(now+.002,tick+1)).mode,Go2Mode.DAMPING)
        self.assertFalse(test.gait_restart.active)

    def test_c_during_recovery_takes_priority_and_never_resumes_left(self):
        test,actor,state=self.trial();pressed=False
        for tick in range(8001):
            now=1+tick*.002
            if tick%10==0 and now>=9.:test.feed_key('a',now)
            if test.gait_restart.active and not pressed:
                test.feed_key('c',now);pressed=True
                deadline=test.started+test.lower_begin
            command=test.output(now,state(now,tick))
            self.assertNotEqual(command.mode,Go2Mode.DAMPING,command.reason)
            if pressed:
                np.testing.assert_array_equal(test.controller.command,0.)
                self.assertIsNone(test.policy_reentry_at)
                if now>deadline+.05:break
        self.assertTrue(pressed)
        self.assertEqual(test.phase(now)[0],'lower')
        self.assertEqual(command.mode,Go2Mode.FIX_STAND)

    def test_x_during_recovery_cancels_lateral_resume(self):
        test,actor,state=self.trial();pressed=False;ready=False
        for tick in range(8001):
            now=1+tick*.002
            if tick%10==0 and now>=9. and not pressed:test.feed_key('a',now)
            if test.gait_restart.active and not pressed:
                test.feed_key('x',now);pressed=True
            command=test.output(now,state(now,tick))
            self.assertNotEqual(command.mode,Go2Mode.DAMPING,command.reason)
            if pressed:
                np.testing.assert_array_equal(test.controller.command,0.)
                if test.policy_reentry_at is not None and now-test.policy_reentry_at>.6:
                    ready=True;break
        self.assertTrue(pressed and ready)
        self.assertEqual(test.gait_restart.count,1)
        np.testing.assert_array_equal(test.keyboard_command,0.)

    def test_z_during_stand_transition_damps_immediately(self):
        test,actor,state=self.trial()
        for tick in range(8001):
            now=1+tick*.002
            if tick%10==0 and now>=9.:test.feed_key('a',now)
            command=test.output(now,state(now,tick))
            if test.gait_restart.active:
                test.stop(now,'主终端Z请求停止')
                stopped=test.output(now+.002,state(now+.002,tick+1))
                self.assertEqual(stopped.mode,Go2Mode.DAMPING)
                np.testing.assert_array_equal(stopped.kp_sdk,0.)
                return
        self.fail('没有进入待验证的恢复阶段')

    def test_stand_recovery_keeps_the_existing_two_rad_s_limit(self):
        test,actor,state=self.trial()
        for tick in range(8001):
            now=1+tick*.002
            if tick%10==0 and now>=9.:test.feed_key('a',now)
            command=test.output(now,state(now,tick))
            if test.gait_restart.active:
                bad=state(now+.002,tick+1);bad.joint_vel_sdk[0]=2.1
                stopped=test.output(now+.002,bad)
                self.assertEqual(stopped.mode,Go2Mode.DAMPING)
                self.assertIn('2rad/s',stopped.reason)
                return
        self.fail('没有进入恢复阶段')

    def test_supported_cli_propagates_flag_and_rejects_pi_combination(self):
        arguments=entry.commission_arguments('all','eno1',Path('zero.jsonl'))+['--gait-restart']
        with patch('sys.stdin.isatty',return_value=True):
            args=commission.parse_args(arguments)
        self.assertTrue(args.gait_restart and args.continuous)
        with self.assertRaises(SystemExit):
            commission.parse_args(arguments+['--velocity-feedback','pi','--velocity-frame','body','--velocity-readiness','shadow.jsonl'])

    def test_cli_rejects_untested_pi_combination_and_short_budget(self):
        for args in [ ['--check','--gait-restart'], ['--check','--gait-restart','--continuous'] ]:
            with self.assertRaises(SystemExit):commission.parse_args(args)


if __name__=='__main__':unittest.main()
