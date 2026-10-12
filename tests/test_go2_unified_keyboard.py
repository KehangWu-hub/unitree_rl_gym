"""统一方向切换、共享预算与日志归零的回归检查。"""
import copy
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from deploy.deploy_real import go2_commission as commission
from deploy.deploy_real import go2_keyboard as entry
from deploy.deploy_real.deploy_real_go2 import Go2RealConfig

ROOT = Path(__file__).resolve().parents[1]


class UnifiedKeyboardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Go2RealConfig.load(ROOT / 'deploy/deploy_real/configs/go2.yaml', str(ROOT))

    def trial(self, motion_seconds=None, walk_speed=.3, continuous=False):
        with tempfile.NamedTemporaryFile(mode='w') as log, patch.object(commission, 'assess_policy', return_value={'result':'zero_velocity_policy_trial_passed'}):
            log.write('{"mode":"policy","feedback":{"policy_entry_blend_seconds":0.5,"policy_pd_limit_nm":14,"policy_feedback_limit_nm":16}}')
            log.flush()
            trial = commission.PolicyCheck(self.config, log.name, policy=lambda obs:torch.zeros((1,12)),
                    forward=True, forward_speed=.4, keyboard=True, keyboard_axis='all', turn_forward_speed=.3, keyboard_motion_seconds=motion_seconds, walk_speed=walk_speed, action_step_mode=None, continuous=continuous)
        trial.started = 0.
        trial.velocity_profile(8.)
        return trial

    def hold(self, trial, key, start, seconds=.4):
        trial.feed_key(key, start)
        command = None
        for i in range(1, int(round(seconds/.02))+1):
            now = start+i*.02
            trial.feed_key(key, now)
            command = trial.velocity_profile(now)
        return command

    def test_walk_speed_changes_only_w_with_same_slew_and_stop(self):
        for key, target in [('w',[.5,0,0]),('s',[-.3,0,0]),('a',[0,.3,0]),('d',[0,-.3,0]),('q',[.3,0,.3]),('e',[.3,0,-.3])]:
            with self.subTest(key=key):
                trial=self.trial(10, .5)
                self.assertAlmostEqual(trial.controller.action_step_limit,.64)
                self.assertAlmostEqual(trial.policy_velocity_limit,20.07)
                previous=np.zeros(3)
                for i in range(1,31):
                    now=8+i*.02;trial.feed_key(key,now);command=trial.velocity_profile(now)
                    self.assertLessEqual(float(np.max(abs(command-previous))),.020001)
                    previous=command
                np.testing.assert_allclose(command,target,atol=1e-6)
                trial.feed_key('x',8.61)
                np.testing.assert_array_equal(trial.velocity_profile(8.62),0)
                self.assertEqual((trial.policy_pd_limit,trial.torque_limit),(14,16))
        trial=self.trial(10,.5);self.hold(trial,'w',8.,.6)
        trial.feed_key('q',8.62)
        np.testing.assert_array_equal(trial.velocity_profile(8.62),0)
        np.testing.assert_allclose(self.hold(trial,'q',8.62),[.3,0,.3],atol=1e-6)

    def test_walk_speed_is_explicit_all_only_and_cli_propagates(self):
        with patch('sys.stdin.isatty',return_value=True):
            args=commission.parse_args(entry.commission_arguments('all','eno1',Path('zero.jsonl'),10,.5,'limit'))
        self.assertEqual(args.walk_speed,.5)
        self.assertEqual(args.action_step_mode,'limit')
        with self.assertRaisesRegex(ValueError,'action-step-mode'):
            entry.commission_arguments('all','eno1',Path('zero.jsonl'),10,.5,'reject')
        self.assertEqual(args.duration,35)
        for mode in ('zero','walk-turn','backward','lateral'):
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                entry.commission_arguments(mode,'eno1',Path('zero.jsonl'),4,.5)
        for speed in (.2,.4,.6,float('nan')):
            with self.subTest(speed=speed),self.assertRaises(ValueError):self.trial(10,speed)
        config=copy.deepcopy(self.config);config.data['command_limits'][0]=.3
        with patch.object(self,'config',config),self.assertRaisesRegex(ValueError,'配置指令上限'):
            self.trial(10,.5)

    def test_every_motion_entry_defaults_to_limit_independent_of_speed(self):
        for mode in ('all', 'walk-turn', 'backward', 'lateral'):
            with self.subTest(mode=mode), patch('sys.stdin.isatty', return_value=True):
                args = commission.parse_args(entry.commission_arguments(mode, 'eno1', Path('zero.jsonl')))
                self.assertEqual(args.action_step_mode, 'limit')
        for speed in (.3, .5):
            trial = self.trial(10, speed)
            self.assertEqual(trial.controller.action_step_mode, 'limit')
            np.testing.assert_allclose(trial.policy_velocity_limits, [25.,25.,20.07]*4)
            self.assertEqual(trial.controller.policy_pd_limit, 14.)
        with patch('sys.stdin.isatty', return_value=True):
            args = commission.parse_args(entry.commission_arguments('zero', 'eno1', Path('stand.jsonl')))
            self.assertEqual(args.action_step_mode, 'reject')

    def test_model_velocity_ceiling_honors_lower_configuration(self):
        config=copy.deepcopy(self.config)
        self.assertAlmostEqual(commission.model_joint_velocity_ceiling(config),20.07)
        config.data['safety']['max_joint_velocity']=12.
        self.assertEqual(commission.model_joint_velocity_ceiling(config),12.)
        config.data['safety']['max_joint_velocity']=float('nan')
        with self.assertRaises(ValueError):commission.model_joint_velocity_ceiling(config)

    def test_walk_speed_assessor_keeps_reverse_lateral_and_turn_limits(self):
        rows=self.rows()
        for r in rows:
            r['feedback']['forward_speed_mps']=.5
            r['feedback']['policy_action_processing']={'mode':'limit','step_limit':.64,'previous_action':[0.]*12,'raw_action':[0.]*12,'applied_action':[0.]*12}
        # Replace the first moving command with the explicit pure-forward candidate.
        idx=next(i for i,r in enumerate(rows) if r['velocity_command'][0]>0 and r['velocity_command'][2]==0)
        rows[idx]['velocity_command']=[.5,0,0]
        self.assertEqual(commission.assess_keyboard(rows,self.config)['result'],'keyboard_unified_and_zero_log_passed')
        for command in ([.51,0,0],[-.5,0,0],[0,.5,0],[.5,0,.3],[.5,0,.5]):
            bad=copy.deepcopy(rows);bad[idx]['velocity_command']=command
            self.assertEqual(commission.assess_keyboard(bad,self.config)['result'],'inconclusive')
        for field,value in [('step_limit',1.),('applied_action',[.7]*12),('raw_action',[4.1]*12)]:
            bad=copy.deepcopy(rows);bad[idx]['feedback']['policy_action_processing'][field]=value
            self.assertEqual(commission.assess_keyboard(bad,self.config)['result'],'inconclusive')
        bad=copy.deepcopy(rows);bad[idx]['feedback']['forward_speed_mps']=.3
        self.assertEqual(commission.assess_keyboard(bad,self.config)['result'],'inconclusive')

    def test_ten_second_profile_preserves_protection_and_shares_budget(self):
        trial=self.trial(10)
        self.assertEqual((trial.keyboard_window_end,trial.keyboard_budget,trial.duration),(26,10,35))
        self.assertEqual((trial.policy_pd_limit,trial.torque_limit,trial.keyboard_timeout),(14,16,.6))
        for i in range(580):
            now=8.02+i*.02
            trial.feed_keys(['w' if i<150 else 'a' if i<300 else 'q'],now)
            command=trial.velocity_profile(now)
        self.assertTrue(10<=trial.keyboard_used<=10.03)
        np.testing.assert_array_equal(command,0)
        trial.feed_key('d',20.)
        self.assertFalse(trial.keyboard_event['accepted'])
        used=trial.keyboard_used;trial.feed_key('x',20.01)
        self.assertEqual(trial.keyboard_used,used)
        self.assertTrue(trial.keyboard_event['accepted'])
        trial=self.trial(10); self.hold(trial,'w',25.)
        np.testing.assert_array_equal(trial.velocity_profile(26.),0)
        trial.feed_key('w',26.)
        self.assertEqual(trial.keyboard_event['reason'],'outside_window')

    def test_ten_second_cli_selects_fixed_duration_and_rejects_other_modes(self):
        with patch('sys.stdin.isatty',return_value=True):
            args=commission.parse_args(entry.commission_arguments('all','eno1',Path('zero.jsonl'),10))
        self.assertEqual(args.duration,35)
        self.assertEqual(args.keyboard_motion_seconds,10)
        for mode in ('zero','walk-turn','backward','lateral'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                entry.commission_arguments(mode,'eno1',Path('zero.jsonl'),10)
        arguments=entry.commission_arguments('all','eno1',Path('zero.jsonl'),10)
        for extra in (['--duration','29'],['--duration','60']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                commission.parse_args(arguments+extra)

    def test_ten_second_log_profile_uses_new_window_and_budget(self):
        rows=self.rows()
        for r in rows:
            f=r['feedback'];f['stand_profile']='keyboard35'
            f['keyboard_motion_budget_seconds']=10.;f['keyboard_window_end_seconds']=26.
        rows[-3]['time']=27.9;rows[-3]['feedback']['test_elapsed']=27.9
        rows[-2]['time']=34.99;rows[-2]['feedback']['test_elapsed']=34.99
        rows[-1]['time']=35.;rows[-1]['feedback']['test_elapsed']=35.
        self.assertEqual(commission.assess_keyboard(rows,self.config)['result'],'keyboard_unified_and_zero_log_passed')
        bad=copy.deepcopy(rows);bad[2]['feedback']['keyboard_motion_budget_seconds']=100
        self.assertEqual(commission.assess_keyboard(bad,self.config)['result'],'inconclusive')
        bad=copy.deepcopy(rows);bad[2]['feedback']['keyboard_forward_seconds']=10.1
        self.assertEqual(commission.assess_keyboard(bad,self.config)['result'],'inconclusive')
        bad=copy.deepcopy(rows);bad[2]['velocity_command']=[0,0,.3]
        self.assertEqual(commission.assess_keyboard(bad,self.config)['result'],'inconclusive')

    def test_default_entry_uses_unified_profile_and_same_torque_and_timing(self):
        with patch('sys.stdin.isatty', return_value=True):
            args = commission.parse_args(entry.commission_arguments('all','eno1',Path('zero.jsonl')))
        trial = self.trial()
        self.assertEqual(args.keyboard_axis,'all')
        self.assertTrue(args.continuous)
        self.assertTrue(np.isinf(args.duration))
        self.assertEqual(args.publish_hz,500)
        self.assertEqual((trial.policy_pd_limit,trial.torque_limit),(14,16))
        self.assertEqual((trial.keyboard_window_end,trial.keyboard_budget,trial.keyboard_timeout),(20,4,.6))
        self.assertEqual(trial.settle_tracking_tolerance,.15)

    def test_continuous_control_runs_beyond_old_deadlines_and_user_stop_latches(self):
        from deploy.deploy_real.deploy_real_go2 import Go2State, Go2Mode
        trial = self.trial(walk_speed=.5, continuous=True)
        self.assertTrue(np.isinf(trial.keyboard_budget))
        self.assertTrue(np.isinf(trial.duration))
        q = self.config.default_joint_pos[self.config.sdk_to_policy].copy()
        def state(now):
            st = Go2State(now,np.array([1.,0,0,0]),np.zeros(3),q.copy(),np.zeros(12))
            st.feedback = {'joint_tau_est_sdk':[0.]*12,'state_tick':int(now*500)}
            return st
        trial.controller.request_fix_stand(0., state(0.))
        trial.hold_samples=50; trial.hold_first=6.; trial.hold_last=6.9
        for i in range(26001):
            now=8.+i*.002
            if i%10==0:
                trial.feed_key('w' if now<35. else 'a' if now<45. else 'q',now)
            command = trial.output(now,state(now))
            self.assertEqual(command.mode,Go2Mode.POLICY,command.reason)
        self.assertGreater(trial.keyboard_used,50.)
        trial.feed_key('x',60.001)
        trial.output(60.02,state(60.02))
        np.testing.assert_array_equal(trial.controller.command,0.)
        trial.feed_key('s',60.021)
        trial.output(60.022,state(60.022))
        self.assertTrue(trial.keyboard_event['accepted'])
        trial.stop(60.03,'主终端Z请求停止')
        self.assertEqual(trial.output(60.032,state(60.032)).mode,Go2Mode.DAMPING)
        trial.feed_key('w',60.04)
        self.assertEqual(trial.output(60.042,state(60.042)).mode,Go2Mode.DAMPING)

    def test_c_zeroes_then_lowers_once_and_z_can_interrupt(self):
        from deploy.deploy_real.deploy_real_go2 import Go2State, Go2Mode
        limited=self.trial(4);limited.policy_command=object()
        self.assertFalse(limited.request_normal_exit(9.))
        trial=self.trial(walk_speed=.5,continuous=True)
        q=self.config.default_joint_pos[self.config.sdk_to_policy].copy()
        def state(now,q_value=q):
            st=Go2State(now,np.array([1.,0,0,0]),np.zeros(3),q_value.copy(),np.zeros(12))
            st.feedback={'joint_tau_est_sdk':[0.]*12,'state_tick':int(now*500)}
            return st
        trial.controller.request_fix_stand(0.,state(0.));trial.hold_samples=50;trial.hold_first=6.;trial.hold_last=6.9
        trial.output(8.,state(8.))
        trial.feed_keys(['c','w'],8.001)
        self.assertIsNotNone(trial.normal_exit)
        self.assertEqual(trial.lower_begin,10.001)
        trial.output(8.02,state(8.02))
        np.testing.assert_array_equal(trial.controller.command,0.)
        duration=trial.duration
        trial.feed_key('c',8.021);trial.feed_key('a',8.022)
        self.assertEqual(trial.duration,duration)
        self.assertFalse(trial.keyboard_event['accepted'])
        now=8.022
        while now<trial.duration+.01:
            now+=.002
            actual=q if now<trial.lower_begin or trial.exit_start is None else trial.profile(now)[0]
            command=trial.output(now,state(now,actual))
        self.assertEqual(command.mode,Go2Mode.DAMPING)
        self.assertEqual(trial.reason,'测试时长结束')
        # 正常降低可随时被紧急停止覆盖。
        trial=self.trial(continuous=True);trial.policy_command=object()
        self.assertTrue(trial.request_normal_exit(9.))
        trial.stop(9.1,'主终端Z请求停止')
        self.assertEqual(trial.output(9.102,state(9.102)).mode,Go2Mode.DAMPING)

    def test_continuous_cli_propagates_and_rejects_conflicting_budget(self):
        args_list=entry.commission_arguments('all','eno1',Path('zero.jsonl'),walk_speed=.5)
        with patch('sys.stdin.isatty',return_value=True):
            args=commission.parse_args(args_list)
        self.assertTrue(args.continuous)
        self.assertIsNone(args.keyboard_motion_seconds)
        for extra in (['--keyboard-motion-seconds','10'],['--duration','35']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                commission.parse_args(args_list+extra)
        with self.assertRaises(ValueError):self.trial(10,continuous=True)

    def test_continuous_log_requires_manual_exit_and_does_not_claim_lowering(self):
        rows=self.rows()
        for r in rows:
            r['feedback'].update(stand_profile='keyboard_continuous',keyboard_continuous=True,
                 keyboard_motion_budget_seconds=None,keyboard_window_end_seconds=None)
        rows[-1]['reason']='主终端Z请求停止'
        result=commission.assess_keyboard(rows,self.config)
        self.assertEqual(result['result'],'keyboard_continuous_manual_stop_log_passed')
        self.assertFalse(result['physical_translation_and_stop_confirmed'])
        self.assertTrue(result['active_x_log_confirmed'])
        for r in rows:
            if (r['feedback'].get('keyboard_event') or {}).get('key')=='x':
                r['feedback'].pop('keyboard_event')
        self.assertEqual(commission.assess_keyboard(rows,self.config)['result'],'keyboard_continuous_manual_stop_log_passed')
        for reason in ('测试时长结束','策略试跑姿态超限','主程序心跳超时'):
            bad=copy.deepcopy(rows);bad[-1]['reason']=reason
            self.assertEqual(commission.assess_keyboard(bad,self.config)['result'],'inconclusive')

    def test_all_keys_have_expected_direction_and_qe_keep_forward_component(self):
        for key,target in [('w',[.3,0,0]),('s',[-.3,0,0]),('a',[0,.3,0]),('d',[0,-.3,0]),('q',[.3,0,.3]),('e',[.3,0,-.3])]:
            with self.subTest(key=key):
                trial=self.trial()
                np.testing.assert_allclose(self.hold(trial,key,8.),target,atol=1e-6)
                self.assertEqual(trial.settle_tracking_tolerance,.25 if key in 'sad' else .15)

    def test_switches_zero_before_accelerating_without_stacking_keys(self):
        trial=self.trial()
        self.hold(trial,'w',8.)
        for i,key in enumerate(('a','q','e','s','w','d')):
            start=8.4+i*.42
            trial.feed_key(key,start+.001)
            np.testing.assert_array_equal(trial.velocity_profile(start+.02),0)
            command=self.hold(trial,key,start+.02,seconds=.4)
            self.assertGreater(float(np.max(np.abs(command))),.29)
            self.assertLessEqual(np.count_nonzero(command),2 if key in 'qe' else 1)

    def test_stop_has_priority_over_direction_repeat_in_same_terminal_batch(self):
        for keys in (['x','w'],['q',' ','e'],['a','x','d']):
            with self.subTest(keys=keys):
                trial=self.trial();self.hold(trial,'q',8.)
                trial.feed_keys(keys,8.42)
                np.testing.assert_array_equal(trial.velocity_profile(8.44),0)
                self.assertIn(trial.keyboard_event['key'],('x',' '))
        trial=self.trial()
        trial.feed_keys(['w','a','e'],8.)
        np.testing.assert_allclose(self.hold(trial,'e',8.),[.3,0,-.3],atol=1e-6)

    def test_budget_is_shared_across_directions_and_turn_not_double_counted(self):
        trial=self.trial()
        for i in range(240):
            now=8.02+i*.02
            trial.feed_key('q' if i<100 else 'd',now)
            command=trial.velocity_profile(now)
        self.assertTrue(4 <= trial.keyboard_used <= 4.03)
        np.testing.assert_array_equal(command,0)
        trial.feed_key('w',13.)
        self.assertFalse(trial.keyboard_event['accepted'])
        self.assertEqual(trial.keyboard_event['reason'],'budget_exhausted')
        trial.feed_key('x',13.01)
        self.assertTrue(trial.keyboard_event['accepted'])

    def test_zero_timeout_and_window_end_clear_every_axis(self):
        for key in ('q','a','s'):
            with self.subTest(key=key):
                trial=self.trial()
                self.hold(trial,key,8.)
                trial.feed_key('x',8.401)
                np.testing.assert_array_equal(trial.keyboard_command,0)
                np.testing.assert_array_equal(trial.velocity_profile(8.42),0)
                self.hold(trial,key,8.42)
                np.testing.assert_array_equal(trial.velocity_profile(9.43),0)
                trial.feed_key(key,19.7)
                trial.velocity_profile(19.72)
                np.testing.assert_array_equal(trial.velocity_profile(20.),0)
                trial.feed_key(key,20.)
                self.assertFalse(trial.keyboard_event['accepted'])
        trial=self.trial(); trial.started=None
        trial.feed_key('w',1.)
        self.assertFalse(trial.keyboard_event['accepted'])

    def test_turn_switch_command_trace_matches_legacy_yaw(self):
        trial=self.trial()
        with tempfile.NamedTemporaryFile(mode='w') as log, patch.object(commission,'assess_policy',return_value={'result':'zero_velocity_policy_trial_passed'}):
            log.write('{"mode":"policy","feedback":{"policy_entry_blend_seconds":0.5,"policy_pd_limit_nm":14,"policy_feedback_limit_nm":16}}')
            log.flush()
            legacy=commission.PolicyCheck(self.config,log.name,policy=lambda obs:torch.zeros((1,12)),
                    forward=True,forward_speed=.4,keyboard=True,keyboard_axis='yaw',turn_forward_speed=.3)
        legacy.started=0.; legacy.velocity_profile(8.)
        for i in range(250):
            now=8.02+i*.02
            key='q' if i<40 else 'e' if i<90 else 'x'
            trial.feed_key(key,now);legacy.feed_key(key,now)
            np.testing.assert_allclose(trial.velocity_profile(now),legacy.velocity_profile(now),atol=1e-7,rtol=0)
        self.assertAlmostEqual(trial.keyboard_used,legacy.keyboard_used)

    def test_unified_profile_rejects_higher_limits_and_in_place_turning(self):
        base=['eno1','--forward-check','ready','--keyboard','--forward-speed','0.4','--keyboard-axis','all','--start-standing','--confirm-standing','--release-mode','--allow-body-movement','--allow-policy-movement']
        for extra in ([],['--turn-forward-speed','0.3','--yaw-speed','0.5']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): commission.parse_args(base+extra)

    def rows(self):
        q=[-.0554,1.2737,-2.8184,.0422,1.270,-2.8053,-.3279,1.2945,-2.7995,.3667,1.3079,-2.8213]
        samples=[(0,'fix_stand',[0,0,0],None),(7,'policy',[0,0,0],None),
                 (8.2,'policy',[.2,0,0],('w',1)),(8.4,'policy',[0,.1,0],('a',2)),
                 (8.6,'policy',[.2,0,-.2],('e',3)),(8.98,'policy',[.2,0,-.2],('e',3)),
                 (9,'policy',[0,0,0],('x',4)),(21.9,'policy',[0,0,0],('x',4)),
                 (28.99,'fix_stand',[0,0,0],None),(29,'damping',[0,0,0],None)]
        rows=[]
        for elapsed,mode,v,event in samples:
            feedback={'stand_profile':'keyboard29','test_elapsed':elapsed,'keyboard_axis':'all',
                      'keyboard_forward_seconds':min(max(elapsed-8,0),1),'yaw_speed_rad_s':.3,
                      'lateral_speed_mps':.3,'forward_speed_mps':.3,'turn_forward_speed_mps':.3}
            if event: feedback['keyboard_event']={'key':event[0],'sequence':event[1],'elapsed':9 if event[0]=='x' else elapsed,'accepted':True}
            rows.append({'time':elapsed,'mode':mode,'reason':'测试时长结束' if elapsed==29 else '',
                         'velocity_command':v,'joint_pos_sdk':q,'joint_vel_sdk':[0]*12,
                         'quaternion_wxyz':[1,0,0,0],'feedback':feedback})
        return rows

    def test_analyzer_accepts_float32_limit_from_real_recorder(self):
        rows=self.rows()
        for row in rows:
            row['feedback']['forward_speed_mps']=float(np.float32(.3))
            row['feedback']['lateral_speed_mps']=float(np.float32(.3))
        self.assertEqual(commission.assess_keyboard(rows,self.config)['result'],'keyboard_unified_and_zero_log_passed')
        rows[1]['feedback']['forward_speed_mps']=.4
        self.assertEqual(commission.assess_keyboard(rows,self.config)['result'],'inconclusive')

    def test_analyzer_handles_mixed_axes_and_requires_active_x_zero(self):
        rows=self.rows()
        self.assertEqual(commission.assess_keyboard(rows,self.config)['result'],'keyboard_unified_and_zero_log_passed')
        for vector in ([.3,.3,0],[0,0,.3],[.3,0,.4],[-.3,0,.3]):
            bad=copy.deepcopy(rows);bad[2]['velocity_command']=vector
            self.assertEqual(commission.assess_keyboard(bad,self.config)['result'],'inconclusive')
        bad=copy.deepcopy(rows)
        for row in bad: row['feedback'].pop('keyboard_event',None)
        self.assertEqual(commission.assess_keyboard(bad,self.config)['result'],'inconclusive')
        bad=copy.deepcopy(rows);bad[6]['velocity_command']=[0,.1,0]
        self.assertEqual(commission.assess_keyboard(bad,self.config)['result'],'inconclusive')


if __name__ == '__main__':
    unittest.main()
