"""Exercise optional velocity feedback without DDS or a robot."""
import json
import os
import time
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from deploy.deploy_real.go2_velocity_feedback import (
    SportVelocitySource, KinematicVelocitySource, VelocityFeedback, body_velocity, validate_feedback_readiness,
)
from deploy.deploy_real import go2_keyboard, go2_commission
from deploy.deploy_real.deploy_real_go2 import Go2State, Go2RealConfig, Go2Mode
import torch


def message(stamp=1, velocity=(.1, 0, 0), yaw=0.):
    return types.SimpleNamespace(stamp=types.SimpleNamespace(sec=stamp, nanosec=0),
                                 velocity=velocity, yaw_speed=yaw, error_code=0)


def state():
    return types.SimpleNamespace(quaternion_wxyz=[1., 0, 0, 0], angular_velocity=[0., 0, 0])


class VelocityFeedbackTest(unittest.TestCase):
    def make(self, mode='pi'):
        obj=VelocityFeedback(mode, [.5, .3, .5], 'body')
        obj.source.snapshot=lambda now: dict(received=now, stamp_ns=int(now*1e9), age_seconds=0.,
                                            velocity_raw=[0., 0., 0.], yaw_speed_raw=0.)
        return obj

    def kinematic_state(self, now=1., tick=1000):
        root=Path(__file__).resolve().parents[1]
        config=Go2RealConfig.load(go2_keyboard.CONFIG,str(root))
        sample=Go2State(now,np.array([1.,0,0,0]),np.zeros(3),
            config.default_joint_pos[config.sdk_to_policy].copy(),np.zeros(12))
        sample.feedback={'state_tick':tick}
        return config,sample

    def test_kinematic_source_uses_lowstate_and_repeated_ticks_expire(self):
        config,sample=self.kinematic_state()
        source=KinematicVelocitySource(config)
        source.observe(sample,1.)
        np.testing.assert_array_equal(source.snapshot(1.)['velocity_raw'],0.)
        sample.timestamp=1.08
        source.observe(sample,1.08)
        self.assertEqual(source.snapshot(1.08)['received'],1.)
        with self.assertRaisesRegex(ValueError,'超时'):source.snapshot(1.11)
        sample.feedback['state_tick']=999
        with self.assertRaisesRegex(ValueError,'倒退'):source.observe(sample,1.08)

    def test_kinematic_shadow_is_identical_and_pi_rejects_bad_sensor(self):
        for mode in ['shadow','pi']:
            config,sample=self.kinematic_state()
            obj=VelocityFeedback(mode,[.5,.4,.5],'body',source='kinematic',config=config,speed_profile='uniform')
            value=obj.apply([.5,0,0],sample,1.)
            self.assertTrue(obj.audit['valid'])
            self.assertEqual(obj.audit['source'],'kinematic')
            if mode=='shadow':np.testing.assert_array_equal(value,[.5,0,0])
            sample.timestamp=1.02;sample.feedback['state_tick']=1020
            sample.joint_vel_sdk[0]=np.nan
            if mode=='pi':
                with self.assertRaises(ValueError):obj.apply([.5,0,0],sample,1.02)
                np.testing.assert_array_equal(obj.pi.integral,0.)
            else:
                np.testing.assert_array_equal(obj.apply([.5,0,0],sample,1.02),[.5,0,0])
                self.assertFalse(obj.audit['valid'])

    def test_uniform_pi_preserves_targets_and_bounded_compensation(self):
        obj=VelocityFeedback('pi',[.5,.4,.5],'body',speed_profile='uniform')
        obj.source.snapshot=lambda now:dict(received=now,stamp_ns=int(now*1e9),age_seconds=0.,velocity_raw=[0,0,0],yaw_speed_raw=0.)
        now=1.
        for target in [[.5,0,0],[-.5,0,0],[0,.4,0],[0,-.4,0],[.5,0,.5],[.5,0,-.5]]:
            for _ in range(70):
                now+=.02;value=obj.apply(target,state(),now)
                self.assertTrue(np.all(abs(value)<=np.array([.6,.5,.6])+1e-7))
            np.testing.assert_allclose(obj.audit['candidate_target'],target)
            expected=np.asarray(target)+np.sign(target)*.1
            np.testing.assert_allclose(value,expected,atol=1e-7)
            now+=.02;np.testing.assert_array_equal(obj.apply([0,0,0],state(),now),0.)
        argv=go2_keyboard.commission_arguments('all','eno1',Path('zero.jsonl'),walk_speed=.5,left_speed=.4,speed_profile='uniform')
        with patch('sys.stdin.isatty',return_value=True):
            args=go2_commission.parse_args(argv+['--velocity-feedback','shadow','--velocity-source','kinematic'])
        self.assertEqual((args.velocity_source,args.velocity_frame),('kinematic','body'))
        with patch('sys.stdin.isatty',return_value=True):
            args=go2_commission.parse_args(argv+['--velocity-feedback','pi','--velocity-source','kinematic','--velocity-readiness','observation.jsonl'])
        self.assertEqual(args.velocity_feedback,'pi')

    def test_world_to_body_at_quarter_turn(self):
        q=[np.sqrt(.5), 0, 0, np.sqrt(.5)]
        np.testing.assert_allclose(body_velocity([0, .3, 0],q,'world'),[.3,0,0],atol=1e-15)
        np.testing.assert_array_equal(body_velocity([0,.3,0],q,'body'),[0,.3,0])

    def test_duplicate_and_out_of_order_stamp_cannot_refresh_source(self):
        source=SportVelocitySource()
        with patch('deploy.deploy_real.go2_velocity_feedback.time.monotonic',return_value=10.):
            source.callback(message(stamp=2))
        with patch('deploy.deploy_real.go2_velocity_feedback.time.monotonic',return_value=10.09):
            source.callback(message(stamp=2));source.callback(message(stamp=1))
        self.assertEqual(source.snapshot(10.09)['received'],10.)
        with self.assertRaisesRegex(ValueError,'超时'):
            source.snapshot(10.11)

    def test_invalid_message_invalidates_prior_good_sample(self):
        source=SportVelocitySource()
        with patch('deploy.deploy_real.go2_velocity_feedback.time.monotonic',return_value=10.):
            source.callback(message());source.callback(message(stamp=2,velocity=[float('nan'),0,0]))
        with self.assertRaisesRegex(ValueError,'无效'):
            source.snapshot(10.01)

    def test_shadow_is_identical_even_if_source_missing_or_frame_unknown(self):
        for frame in [None,'body','world']:
            obj=VelocityFeedback('shadow',[.5,.3,.5],frame)
            target=np.array([.3,0,-.3])
            np.testing.assert_array_equal(obj.apply(target,state(),1.),target)
            self.assertFalse(obj.audit['valid'])

    def test_shadow_valid_feedback_never_changes_target(self):
        obj=self.make('shadow')
        for n in range(200):
            np.testing.assert_array_equal(obj.apply([.3,0,.3],state(),1+n*.02),[.3,0,.3])
        self.assertTrue(obj.audit['valid'])

    def test_saturated_pi_is_bounded_and_x_clears_integral(self):
        obj=self.make()
        previous=np.zeros(3)
        for n in range(500):
            value=obj.apply([0,.3,0],state(),1+n*.02)
            self.assertTrue(np.all(abs(value)<=np.array([.5,.3,.5])+1e-9))
            self.assertTrue(np.all(abs(value-previous)<=.020001))
            previous=value
        self.assertLess(abs(obj.pi.integral[1]),1.)
        np.testing.assert_array_equal(obj.apply([0,0,0],state(),11.),0.)
        np.testing.assert_array_equal(obj.pi.integral,0.)
        np.testing.assert_array_equal(obj.pi.command,0.)

    def test_direction_change_cannot_reuse_accumulated_integral(self):
        obj=self.make()
        for n in range(100):obj.apply([.3,0,0],state(),1+n*.02)
        value=obj.apply([0,.3,0],state(),3.)
        self.assertLessEqual(np.max(abs(value)),.020001)
        self.assertLess(np.max(abs(obj.pi.integral)),.001)

    def test_missing_feedback_and_yaw_disagreement_fail_pi(self):
        obj=self.make()
        obj.source.snapshot=lambda now: (_ for _ in ()).throw(ValueError('速度超时'))
        with self.assertRaisesRegex(ValueError,'超时'):obj.apply([.3,0,0],state(),1.)
        obj=self.make()
        obj.source.snapshot=lambda now:dict(velocity_raw=[0,0,0],yaw_speed_raw=1.)
        with self.assertRaisesRegex(ValueError,'不一致'):obj.apply([.3,0,0],state(),1.)

    def test_pi_requires_explicit_frame(self):
        with self.assertRaisesRegex(ValueError,'坐标系'):
            VelocityFeedback('pi',[.5,.3,.5])

    def test_cli_rejects_pi_without_post_handoff_observation(self):
        with self.assertRaises(SystemExit) as result:
            go2_keyboard.main(['--velocity-feedback','pi','--velocity-frame','body','--check'])
        self.assertEqual(result.exception.code,2)
        with self.assertRaises(SystemExit):
            go2_commission.parse_args(['--check','--velocity-feedback','pi'])

    def test_post_handoff_readiness_rejects_missing_stale_and_wrong_policy(self):
        rows=[]
        for n in range(301):
            rows.append(dict(mode='policy',feedback=dict(test_elapsed=8+n*.02,policy_sha256='p',
                policy_config_sha256='c',velocity_feedback=dict(mode='shadow',valid=True,age_seconds=0.,
                received=100+n*.02,stamp_ns=1000000000+n*20000000,velocity_raw=[.1,0,0]))))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'shadow.jsonl'
            def write():path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
            write()
            self.assertGreaterEqual(validate_feedback_readiness(path,'p','c')['post_handoff_seconds'],5.)
            with self.assertRaisesRegex(ValueError,'来源'):validate_feedback_readiness(path,'p','c',source='kinematic')
            with self.assertRaisesRegex(ValueError,'版本'):validate_feedback_readiness(path,'p','c',estimator_sha256='different')
            with self.assertRaisesRegex(ValueError,'版本不匹配'):validate_feedback_readiness(path,'wrong','c')
            rows[100]['feedback']['velocity_feedback']['valid']=False;write()
            with self.assertRaisesRegex(ValueError,'反馈缺失'):validate_feedback_readiness(path,'p','c')
            rows[100]['feedback']['velocity_feedback']['valid']=True
            for row in rows:row['feedback']['velocity_feedback']['stamp_ns']=1
            write()
            with self.assertRaisesRegex(ValueError,'时间戳推进'):validate_feedback_readiness(path,'p','c')

    def test_real_policy_class_damps_on_feedback_loss_but_shadow_continues(self):
        root=Path(__file__).resolve().parents[1]
        config=Go2RealConfig.load(root/'deploy/deploy_real/configs/go2.yaml',str(root))
        for mode in ['shadow','pi']:
            with self.subTest(mode=mode), tempfile.NamedTemporaryFile(mode='w') as log, \
                 patch.object(go2_commission,'assess_policy',return_value={'result':'zero_velocity_policy_trial_passed'}):
                log.write(json.dumps({'mode':'policy','feedback':{'policy_entry_blend_seconds':.5,
                    'policy_pd_limit_nm':14.,'policy_feedback_limit_nm':16.}}));log.flush()
                trial=go2_commission.PolicyCheck(config,log.name,policy=lambda obs:torch.zeros((1,12)),
                    forward=True,forward_speed=.4,keyboard=True,keyboard_axis='all',turn_forward_speed=.3,continuous=True)
            observer=self.make(mode)
            def sample(now):
                if now>=10.:raise ValueError('注入速度超时')
                return dict(received=now,stamp_ns=int(now*1e9),age_seconds=0.,velocity_raw=[0.,0.,0.],yaw_speed_raw=0.)
            observer.source.snapshot=sample
            trial.velocity_feedback=observer
            q=config.default_joint_pos[config.sdk_to_policy]
            def sample_state(now,tick):
                st=Go2State(now,np.array([1.,0,0,0]),np.zeros(3),q.copy(),np.zeros(12))
                st.feedback={'joint_tau_est_sdk':[0.]*12,'state_tick':tick}
                return st
            trial.arm(1.,sample_state(1.,0))
            for tick in range(4511):
                now=1.+tick*.002
                if now>=9. and tick%10==0:trial.feed_key('w',now)
                command=trial.output(now,sample_state(now,tick))
                if command.mode is Go2Mode.DAMPING:break
            if mode=='pi':
                self.assertEqual(command.mode,Go2Mode.DAMPING)
                self.assertIn('注入速度超时',command.reason)
                np.testing.assert_array_equal(command.kp_sdk,0.)
                self.assertLessEqual(now,10.02)
            else:
                self.assertEqual(command.mode,Go2Mode.POLICY)
                self.assertFalse(observer.audit['valid'])


@unittest.skipUnless(os.environ.get('GO2_DDS_TESTS')=='1','设置GO2_DDS_TESTS=1执行lo速度订阅测试')
class SportVelocityDDSTest(unittest.TestCase):
    def test_real_dds_velocity_stream_and_frozen_publisher_timestamp(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
        ChannelFactoryInitialize(231,'lo')
        source=SportVelocitySource(timeout=.1)
        publisher=ChannelPublisher('rt/sportmodestate',SportModeState_)
        publisher.Init();source.start()
        msg=unitree_go_msg_dds__SportModeState_()
        msg.velocity=[.12,-.03,0.];msg.yaw_speed=.04;msg.stamp.sec=1
        deadline=time.monotonic()+3.;sample=None
        try:
            while time.monotonic()<deadline:
                msg.stamp.nanosec+=10000000
                publisher.Write(msg);time.sleep(.01)
                try:
                    sample=source.snapshot(time.monotonic());break
                except ValueError:pass
            self.assertIsNotNone(sample,'lo SportModeState没有传到速度订阅器')
            np.testing.assert_allclose(sample['velocity_raw'],[.12,-.03,0.],atol=1e-6)
            # Keep receiving DDS frames while the publisher stamp is frozen.
            until=time.monotonic()+.18
            while time.monotonic()<until:
                publisher.Write(msg);time.sleep(.01)
            with self.assertRaisesRegex(ValueError,'超时'):
                source.snapshot(time.monotonic())
        finally:
            source.close();publisher.Close()


if __name__=='__main__':unittest.main()
