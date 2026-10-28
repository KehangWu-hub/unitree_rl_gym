"""Offline replay must distinguish intact, altered, and incomplete field evidence."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from deploy.common.him_qualification import sha256
from deploy.deploy_real.audit_him_sim2real import audit
from deploy.deploy_real.deploy_real_go2 import Go2RealConfig

ROOT=Path(__file__).resolve().parents[1]


class HIMReplayAuditTest(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        layer=torch.nn.Linear(270,12)
        with torch.no_grad():
            layer.weight.fill_(.001)
            layer.bias.copy_(torch.linspace(-.1,.1,12))
        model=torch.jit.trace(layer.eval(),torch.zeros(1,270))
        path=self.root/'policy.pt';model.save(str(path))
        cfg=Go2RealConfig.load(ROOT/'deploy/deploy_real/configs/go2_him_fresh.yaml',str(ROOT))
        self.config=replace(cfg,policy_path=str(path))
        self.rows=[];previous=np.zeros(12,dtype=np.float32);frames=[]
        for i in range(2):
            # Construct the training observation independently of build_observation.
            frame=np.concatenate(([0,0,0],[0,0,0],[0,0,-1],np.zeros(24),previous)).astype(np.float32)
            frames.insert(0,frame)
            observation=np.concatenate(frames+[np.zeros(45,dtype=np.float32)]*(6-len(frames)))
            with torch.inference_mode():raw=model(torch.from_numpy(observation).unsqueeze(0)).numpy()[0]
            target=(cfg.default_joint_pos+raw*.25).astype(np.float32)
            row={'time':8.+i*.02,'state_time':8.+i*.02,'mode':'policy','reason':'',
                 'angular_velocity':[0,0,0],'quaternion_wxyz':[1,0,0,0],
                 'joint_pos_sdk':cfg.default_joint_pos[cfg.sdk_to_policy].tolist(),'joint_vel_sdk':[0]*12,
                 'velocity_command':[0,0,0],'target_pos_sdk':target[cfg.sdk_to_policy].tolist(),
                 'kp_sdk':[20]*12,'kd_sdk':[.5]*12,
                 'feedback':{'test_elapsed':7.+i*.02,'state_tick':1000+i*20,
                             'policy_sha256':sha256(path),'policy_config_sha256':cfg.source_sha256,
                             'policy_inference_count':i+1,'joint_tau_est_sdk':[0]*12,
                             'policy_action_processing':{'raw_action':raw.tolist(),'applied_action':raw.tolist(),'step_limit':.64},
                             'policy_target_processing':{'raw_target_policy':target.tolist()}}}
            self.rows.append(row);previous=raw.copy()
        self.log=self.root/'field.jsonl'

    def replay(self):
        self.log.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        return audit(self.config,self.log)

    def test_independent_two_frame_history_replays_exactly(self):
        report,arrays=self.replay()
        self.assertTrue(report['replay_matched'])
        self.assertEqual(report['inferences_replayed'],2)
        self.assertEqual(arrays['observations'].shape,(2,270))
        np.testing.assert_array_equal(arrays['observations'][0,45:],0.)
        np.testing.assert_array_equal(arrays['observations'][1,45:90],arrays['observations'][0,:45])

    def test_changed_recorded_action_is_not_accepted(self):
        self.rows[0]['feedback']['policy_action_processing']['raw_action'][0]+=.2
        report,_=self.replay()
        self.assertFalse(report['replay_matched'])
        self.assertGreater(report['max_raw_action_error'],.1)

    def test_missing_inference_refuses_to_guess_history(self):
        self.rows[1]['feedback']['policy_inference_count']=3
        with self.assertRaisesRegex(ValueError,'Inference missing'):self.replay()


if __name__=='__main__':unittest.main()
