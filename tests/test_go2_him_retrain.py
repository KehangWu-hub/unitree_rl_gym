"""Independent action-density and deployment-processing checks for fresh HIM training."""
import ast,contextlib,io,math,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch
from torch.distributions import Normal,TransformedDistribution,transforms

from third_party.himloco_rsl_rl.rsl_rl.modules import BoundedHIMActorCritic
from deploy.deploy_real.deploy_real_go2 import Go2RealConfig,Go2Controller
from tests.test_go2_him_training import EXPORT,env_method,file_module

ROOT=Path(__file__).resolve().parents[1]


def method(name):
    path=ROOT/'legged_gym/envs/go2_him/retrain_env.py'
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    scope={'torch':torch};exec(compile(module,str(path),'exec'),scope)
    return scope[name]


class HIMRetrainTest(unittest.TestCase):
    def network(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return BoundedHIMActorCritic(270,238,45,12,actor_hidden_dims=[32,16],critic_hidden_dims=[32,16])

    def test_bounded_action_log_density_matches_torch_transformed_distribution(self):
        model=self.network();history=torch.randn(32,270)*.2
        action=model.act(history)
        self.assertTrue(torch.all(action.abs()<=4))
        reference=TransformedDistribution(model.distribution,[transforms.AffineTransform(0.,.25),
            transforms.TanhTransform(cache_size=1),transforms.AffineTransform(0.,4.)])
        torch.testing.assert_close(model.get_actions_log_prob(action),reference.log_prob(action).sum(-1),rtol=1e-5,atol=1e-5)
        model.entropy.mean().backward()
        self.assertTrue(torch.isfinite(model.log_std.grad).all())

    def test_export_contains_bound_and_equals_inference(self):
        model=self.network().eval();history=torch.randn(8,270)
        with torch.no_grad():
            model.actor[-1].bias.fill_(100.)
            expected=model.act_inference(history)
            self.assertTrue(torch.all(expected.abs()<=4.))
            with tempfile.TemporaryDirectory() as d:
                policy=torch.jit.load(EXPORT.export_him_policy(model,d))
                torch.testing.assert_close(policy(history),expected,rtol=0,atol=0)

    def test_training_direct_actions_match_unrestricted_real_controller(self):
        config=Go2RealConfig.load(ROOT/'deploy/deploy_real/configs/go2_him_fresh.yaml',str(ROOT))
        controller=Go2Controller(config,policy=lambda obs:torch.zeros((1,12)),action_step_mode='limit')
        for request in (np.full(12,4.),np.full(12,-4.),np.linspace(-3,3,12),np.zeros(12)):
            real=controller.process_policy_action(request)
            trained=env_method('_prepare_control_actions')(NS(),torch.tensor(request,dtype=torch.float32)[None])
            np.testing.assert_allclose(trained[0].numpy(),real,atol=1e-6)
            np.testing.assert_allclose(real,request,atol=1e-6)
            controller.last_action=real

    def test_training_direct_target_matches_real_with_only_model_torque_cap(self):
        config=Go2RealConfig.load(ROOT/'deploy/deploy_real/configs/go2_him_fresh.yaml',str(ROOT))
        controller=Go2Controller(config,policy=lambda obs:torch.zeros((1,12)))
        q=config.default_joint_pos.copy();dq=np.linspace(-2,2,12).astype(np.float32)
        action=np.full(12,4.,dtype=np.float32)
        lower=np.asarray(config.safety['joint_lower'],dtype=np.float32);upper=np.asarray(config.safety['joint_upper'],dtype=np.float32)
        env=NS(cfg=NS(control=NS(action_scale=.25)),
            dof_pos_limits=torch.tensor(np.stack([lower,upper],axis=1)),
            default_dof_pos=torch.tensor(q)[None],dof_pos=torch.tensor(q)[None],dof_vel=torch.tensor(dq)[None],
            p_gains=torch.tensor(config.stiffness),d_gains=torch.tensor(config.damping),
            Kp_factors=torch.ones(1,1),Kd_factors=torch.ones(1,1),
            motor_strength_factors=torch.tensor([[1.1]]),
            torque_limits=torch.tensor(config.torque_limits))
        actual=method('_compute_torques')(env,torch.tensor(action)[None])
        target=controller.safety.limit_target(q+action*.25)
        expected_target=controller.safety.limit_pd_target(target,q,dq,config.stiffness,config.damping,14.)
        expected=np.clip(1.1*(config.stiffness*(expected_target-q)-config.damping*dq),-config.torque_limits,config.torque_limits)
        np.testing.assert_allclose(env.joint_pos_target[0].numpy(),expected_target,atol=1e-6)
        np.testing.assert_allclose(actual[0].numpy(),expected,atol=1e-5)
        self.assertGreater(float(actual.abs().max()),14.)
        np.testing.assert_allclose(expected_target,q+action*.25,atol=1e-6)

    def test_balanced_commands_respect_declared_ranges_and_include_zero(self):
        sample=method('_resample_commands')
        sample.__globals__['torch_rand_float']=lambda lo,hi,shape,device:lo+(hi-lo)*torch.rand(shape)
        env=NS(device='cpu',commands=torch.zeros(8192,4),command_ranges={
            'lin_vel_x':[-.5,.5],'lin_vel_y':[-.35,.35],'ang_vel_yaw':[-.5,.5]})
        torch.manual_seed(42);sample(env,torch.arange(8192))
        self.assertTrue(torch.all(env.commands[:,:3].abs()<=torch.tensor([.5,.35,.5])))
        self.assertGreater(int((env.commands[:,:3].abs().sum(1)==0).sum()),500)
        for axis in range(3):
            self.assertTrue(torch.any(env.commands[:,axis]>.1))
            self.assertTrue(torch.any(env.commands[:,axis]<-.1))

    def test_acceptance_rejects_low_pose_and_single_stalled_environment(self):
        acceptance=file_module('him_retrain_acceptance',ROOT/'legged_gym/envs/go2_him/acceptance.py')
        time=np.arange(2500)*.02
        samples=np.zeros((2500,8,9));samples[:,:,0]=time[:,None]
        command=np.where(time<20.,.3,np.where(time<40.,-.3,0.))
        samples[:,:,1]=command[:,None];samples[:,:,4]=command[:,None]
        samples[:,:,7]=.30
        self.assertTrue(acceptance.assess_motion(samples)['passed'])
        low=samples.copy();low[:,:,7]=.25
        self.assertFalse(acceptance.assess_motion(low)['checks']['height_mean'])
        stalled=samples.copy();stalled[:,7,4]=0.
        self.assertFalse(acceptance.assess_motion(stalled)['checks']['linear_tracking'])
        self.assertFalse(acceptance.assess_motion(samples[:500])['passed'])


if __name__=='__main__':unittest.main()
