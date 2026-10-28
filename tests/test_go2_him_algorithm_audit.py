"""Training semantics that must hold before spending GPU time or using a policy."""
import contextlib
import io
from types import SimpleNamespace as NS
import unittest

import torch

from tests.test_go2_him_training import env_method, observation_fixture
from third_party.himloco_rsl_rl.rsl_rl.modules import HIMActorCritic
from third_party.himloco_rsl_rl.rsl_rl.modules.him_estimator import HIMEstimator, sinkhorn
from third_party.himloco_rsl_rl.rsl_rl.algorithms import HIMPPO


class HIMAlgorithmAuditTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def network(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return HIMActorCritic(270,238,45,12,actor_hidden_dims=[32,16],critic_hidden_dims=[32,16])

    def test_reset_does_not_change_pending_ppo_actor_or_critic_observations(self):
        env=observation_fixture()
        env.obs_buf.fill_(.2);env.privileged_obs_buf.fill_(.3)
        algorithm=HIMPPO(self.network(),device='cpu')
        before_actor,before_critic=env.obs_buf.clone(),env.privileged_obs_buf.clone()
        algorithm.act(env.obs_buf,env.privileged_obs_buf)
        env_method('_reset_observation_history')(env,torch.tensor([1]))
        torch.testing.assert_close(algorithm.transition.observations,before_actor,rtol=0,atol=0)
        torch.testing.assert_close(algorithm.transition.critic_observations,before_critic,rtol=0,atol=0)
        self.assertEqual(int(torch.count_nonzero(env.obs_buf[1])),0)
        self.assertEqual(int(torch.count_nonzero(env.privileged_obs_buf[1])),0)

    def test_motor_strength_is_applied_before_physical_torque_clipping(self):
        env=NS(cfg=NS(control=NS(action_scale=.25,hip_reduction=1.,control_type='P')),
            default_dof_pos=torch.zeros(1,12),dof_pos=torch.zeros(2,12),dof_vel=torch.zeros(2,12),
            p_gains=torch.ones(12)*20,d_gains=torch.ones(12)*.5,Kp_factors=torch.ones(2,1),
            Kd_factors=torch.ones(2,1),motor_strength_factors=torch.tensor([[.5],[1.5]]),
            torque_limits=torch.ones(12)*6)
        torque=env_method('_compute_torques')(env,torch.ones(2,12))
        torch.testing.assert_close(torque[:,0],torch.tensor([2.5,6.]))

    def test_initial_joint_position_flag_and_range_are_effective(self):
        method=env_method('_reset_dofs');scope=method.__globals__;calls=[]
        def sample(lower,upper,shape,device):
            calls.append((lower,upper));return torch.full(shape,upper)
        scope.update(torch_rand_float=sample,gymtorch=NS(unwrap_tensor=lambda x:x))
        env=NS(cfg=NS(domain_rand=NS(randomize_initial_joint_pos=False,initial_joint_pos_range=[.8,1.2])),
            dof_pos=torch.zeros(2,12),default_dof_pos=torch.ones(1,12),dof_vel=torch.ones(2,12),
            num_dof=12,device='cpu',gym=NS(set_dof_state_tensor_indexed=lambda *a:None),
            sim=None,dof_state=torch.zeros(24,2))
        method(env,torch.tensor([1]))
        torch.testing.assert_close(env.dof_pos[1],torch.ones(12))
        torch.testing.assert_close(env.dof_pos[0],torch.zeros(12))
        self.assertFalse(calls)
        env.cfg.domain_rand.randomize_initial_joint_pos=True
        method(env,torch.tensor([1]))
        torch.testing.assert_close(env.dof_pos[1],torch.ones(12)*1.2)
        self.assertEqual(calls,[(.8,1.2)])

    def test_target_encoder_uses_successor_state_not_command_and_velocity_label_is_scaled(self):
        estimator=HIMEstimator(6,45)
        history=torch.randn(16,270)*.1
        successor=torch.randn(16,238)*.1
        successor[:,:3]=900.
        successor[:,45:48]=torch.tensor([.8,-.4,.2])
        expected=torch.nn.functional.mse_loss(estimator.encoder(history)[:,:3],successor[:,45:48]).item()
        captured=[]
        handle=estimator.target.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
        estimation,swap=estimator.update(history,successor);handle.remove()
        self.assertAlmostEqual(estimation,expected,places=6)
        torch.testing.assert_close(captured[0][:,:3],successor[:,3:6])
        torch.testing.assert_close(captured[0][:,-3:],successor[:,45:48])
        self.assertFalse(torch.any(captured[0]==900.))
        self.assertTrue(torch.isfinite(torch.tensor([estimation,swap])).all())

    def test_actor_gradient_is_isolated_from_estimator(self):
        model=self.network()
        model.update_distribution(torch.randn(16,270)*.1)
        model.action_mean.square().mean().backward()
        self.assertTrue(any(p.grad is not None and torch.any(p.grad!=0) for p in model.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in model.estimator.parameters()))

    def test_sinkhorn_outputs_detached_finite_sample_assignments(self):
        scores=torch.randn(64,32,requires_grad=True)*.1
        assignments=sinkhorn(scores)
        self.assertFalse(assignments.requires_grad)
        self.assertTrue(torch.isfinite(assignments).all())
        self.assertTrue(torch.all(assignments>=0))
        torch.testing.assert_close(assignments.sum(dim=1),torch.ones(64),rtol=1e-6,atol=1e-6)

    def test_ppo_update_changes_parameters_and_reports_all_minibatch_means(self):
        model=self.network();algorithm=HIMPPO(model,device='cpu',num_learning_epochs=2,num_mini_batches=2)
        algorithm.init_storage(8,4,[270],[238],[12])
        snapshots={group:{name:p.detach().clone() for name,p in module.named_parameters()}
                   for group,module in [('actor',model.actor),('critic',model.critic),('estimator',model.estimator)]}
        obs=torch.randn(8,270)*.1;critic=torch.randn(8,238)*.1
        for _ in range(4):
            algorithm.act(obs,critic)
            successor=torch.randn(8,238)*.1
            algorithm.process_env_step(torch.rand(8),torch.zeros(8,dtype=torch.bool),{},successor)
            obs=torch.randn(8,270)*.1;critic=successor
        algorithm.compute_returns(critic)
        original=model.estimator.update;count=[0]
        def update(*args,**kwargs):
            original(*args,**kwargs);count[0]+=1
            return float(count[0]),10.*count[0]
        model.estimator.update=update
        losses=algorithm.update()
        self.assertTrue(torch.isfinite(torch.tensor(losses)).all())
        self.assertEqual(losses[2:],(2.5,25.))
        for group,module in [('actor',model.actor),('critic',model.critic),('estimator',model.estimator)]:
            self.assertTrue(any(not torch.equal(p,snapshots[group][name]) for name,p in module.named_parameters()),group)


if __name__=='__main__':unittest.main()
