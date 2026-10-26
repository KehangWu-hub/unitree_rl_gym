"""HIM runtime history, safety separation, and physical evidence boundaries."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from deploy.common.go2_policy import PolicyObservationHistory, build_observation, validate_policy_config
from deploy.deploy_real.deploy_real_go2 import Go2Controller, Go2RealConfig, Go2State, Go2Mode
from deploy.deploy_real.go2_commission import PolicyCheck, parse_args
from deploy.deploy_real.go2_keyboard import commission_arguments

ROOT=Path(__file__).resolve().parents[1]


class HIMDeployTest(unittest.TestCase):
    def setUp(self):
        self.config=Go2RealConfig.load(ROOT/'deploy/deploy_real/configs/go2_him.yaml',str(ROOT))

    def test_rejects_incompatible_temporal_contract_and_mapping(self):
        for key,value in [('history_order','oldest_first'),('history_length',5),('num_observations',45)]:
            data=copy.deepcopy(self.config.data);data['policy'][key]=value
            with self.assertRaises(ValueError):validate_policy_config(data)
        data=copy.deepcopy(self.config.data);data['policy']['policy_to_sdk']=list(range(12))
        with self.assertRaises(ValueError):validate_policy_config(data)

    def test_history_order_clipping_reset_and_owned_arrays(self):
        history=PolicyObservationHistory(self.config.data)
        frame=build_observation([1,2,3],[0,0,-1],[.3,-.2,.1],self.config.default_joint_pos,
                                 np.ones(12)*1e5,np.ones(12),self.config.data)
        np.testing.assert_allclose(frame[:9],[.6,-.4,.025,.25,.5,.75,0,0,-1],atol=1e-7)
        np.testing.assert_equal(frame[21:33],100.)
        first=history.append(frame)
        np.testing.assert_equal(first[45:],0.)
        second=history.append(frame+1)
        np.testing.assert_array_equal(second[45:90],frame)
        first[:]=999
        self.assertFalse(np.any(history.frames==999))
        history.reset();np.testing.assert_equal(history.frames,0.)
        with self.assertRaises(ValueError):history.append(np.full(45,np.nan))

    def test_him_previous_action_is_network_output_not_safety_limited_action(self):
        class Actor:
            def __init__(self):self.inputs=[]
            def __call__(self,obs):
                self.inputs.append(obs.clone());return torch.ones((1,12))*1.5
        actor=Actor();controller=Go2Controller(self.config,policy=actor,action_step_mode='limit')
        q=self.config.default_joint_pos
        state=Go2State(1.,np.array([1.,0,0,0]),np.zeros(3),q[self.config.sdk_to_policy],np.zeros(12))
        action,_=controller._policy_action(state,q,np.zeros(12))
        applied=controller.process_policy_action(action)
        controller.last_action=applied
        self.assertLess(float(applied.max()),1.5)
        controller._policy_action(state,q,np.zeros(12))
        np.testing.assert_equal(actor.inputs[-1][0,33:45].numpy(),1.5)
        np.testing.assert_equal(actor.inputs[-1][0,78:90].numpy(),0.)
        controller.reset_policy_state()
        controller._policy_action(state,q,np.zeros(12))
        np.testing.assert_equal(actor.inputs[-1][0,45:].numpy(),0.)
        np.testing.assert_equal(actor.inputs[-1][0,33:45].numpy(),0.)

    def test_policy_reentry_and_emergency_clear_temporal_state(self):
        controller=Go2Controller(self.config,policy=lambda obs:torch.zeros((1,12)))
        controller.observation_history.frames[:]=1;controller.previous_policy_action[:]=2
        controller.fsm.mode=Go2Mode.FIX_STAND;controller.fsm.stand_complete=True
        self.assertTrue(controller.request_policy(1.))
        np.testing.assert_equal(controller.observation_history.frames,0.)
        controller.observation_history.frames[:]=1
        controller.emergency_stop(2.)
        np.testing.assert_equal(controller.observation_history.frames,0.)

    def test_physical_readiness_rejects_simulation_and_other_model(self):
        for row in [dict(simulation_only=True,mode='policy'),dict(mode='policy',feedback={})]:
            with tempfile.NamedTemporaryFile(mode='w') as f, patch('deploy.common.him_qualification.require_sim2sim',return_value={}):
                f.write(json.dumps(row)+'\n');f.flush()
                with self.assertRaises(ValueError):PolicyCheck(self.config,f.name,forward=True)

    def test_keyboard_preserves_explicit_him_config(self):
        path=ROOT/'deploy/deploy_real/configs/go2_him.yaml'
        args=commission_arguments('all','eno1','new_physical.jsonl',config_path=path)
        self.assertEqual(args[args.index('--config')+1],str(path))

    def test_him_zero_cli_uses_safe_action_limit_and_extended_exit(self):
        from unittest.mock import patch
        path=ROOT/'deploy/deploy_real/configs/go2_him.yaml'
        args=commission_arguments('zero','eno1','new_stand.jsonl',config_path=path)
        with patch('sys.stdin.isatty',return_value=True):
            parsed=parse_args(args)
        self.assertEqual(parsed.duration,30.)
        self.assertEqual(parsed.action_step_mode,'limit')

    def test_him_rejects_unvalidated_fast_profile(self):
        with self.assertRaisesRegex(ValueError,'0.30'):
            PolicyCheck(self.config,'unused',forward=True,keyboard=True,keyboard_axis='all',
                        forward_speed=.4,continuous=True,speed_profile='uniform',
                        walk_speed=.5,left_speed=.4,yaw_speed=.5,turn_forward_speed=.5)


if __name__=='__main__':unittest.main()
