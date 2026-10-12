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

    def trial(self):
        with tempfile.NamedTemporaryFile(mode='w') as log, patch.object(commission, 'assess_policy', return_value={'result':'zero_velocity_policy_trial_passed'}):
            log.write('{"mode":"policy","feedback":{"policy_entry_blend_seconds":0.5,"policy_pd_limit_nm":14,"policy_feedback_limit_nm":16}}')
            log.flush()
            trial = commission.PolicyCheck(self.config, log.name, policy=lambda obs:torch.zeros((1,12)),
                    forward=True, forward_speed=.4, keyboard=True, keyboard_axis='all', turn_forward_speed=.3)
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

    def test_default_entry_uses_unified_profile_and_same_torque_and_timing(self):
        with patch('sys.stdin.isatty', return_value=True):
            args = commission.parse_args(entry.commission_arguments('all','eno1',Path('zero.jsonl')))
        trial = self.trial()
        self.assertEqual(args.keyboard_axis,'all')
        self.assertEqual(args.duration,29)
        self.assertEqual(args.publish_hz,500)
        self.assertEqual((trial.policy_pd_limit,trial.torque_limit),(14,16))
        self.assertEqual((trial.keyboard_window_end,trial.keyboard_budget,trial.keyboard_timeout),(20,4,.6))
        self.assertEqual(trial.settle_tracking_tolerance,.15)

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
