"""The upstream flat baseline must not inherit custom retraining settings."""
import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

import torch
import yaml

from third_party.himloco_rsl_rl.rsl_rl.modules import HIMActorCritic
from tests.test_go2_him_training import DEPLOY, EXPORT, file_module, observation_fixture

ROOT = Path(__file__).resolve().parents[1]


def config_scope():
    base = file_module('upstream_cfg_base', ROOT/'legged_gym/envs/base/base_config.py')
    scope = {'BaseConfig': base.BaseConfig}
    for name in ('base_config.py', 'go2_him_config.py', 'upstream_flat_config.py'):
        path = ROOT/'legged_gym/envs/go2_him'/name
        tree = ast.parse(path.read_text())
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, str(path), 'exec'), scope)
    return scope


class UpstreamFlatTest(unittest.TestCase):
    def test_original_sampling_optimizer_and_gaussian_policy_are_restored(self):
        scope = config_scope()
        cfg, ppo = scope['Go2HIMUpstreamFlatCfg'](), scope['Go2HIMUpstreamFlatCfgPPO']()
        self.assertEqual(cfg.env.num_envs, 4096)
        self.assertEqual(ppo.runner.num_steps_per_env, 100)
        self.assertEqual(ppo.runner.policy_class_name, 'HIMActorCritic')
        self.assertEqual(ppo.algorithm.learning_rate, .001)
        self.assertEqual(ppo.algorithm.schedule, 'adaptive')
        self.assertEqual(ppo.algorithm.entropy_coef, .01)
        self.assertEqual(ppo.policy.init_noise_std, 1.)
        self.assertEqual(cfg.normalization.clip_actions, 100.)
        self.assertAlmostEqual(cfg.sim.dt*cfg.control.decimation, .02)
        self.assertFalse(hasattr(ppo.policy, 'action_bound'))
        self.assertFalse(hasattr(cfg.control, 'action_step_limit'))
        self.assertFalse(hasattr(cfg.control, 'policy_pd_limit'))

    def test_rewards_and_command_randomization_inherit_go2_adapter(self):
        scope = config_scope()
        self.assertIs(scope['Go2HIMUpstreamFlatCfg'].rewards, scope['Go2HIMCfg'].rewards)
        self.assertIs(scope['Go2HIMUpstreamFlatCfg'].commands, scope['Go2HIMCfg'].commands)
        self.assertIs(scope['Go2HIMUpstreamFlatCfg'].domain_rand, scope['Go2HIMCfg'].domain_rand)
        cfg = scope['Go2HIMUpstreamFlatCfg']()
        self.assertEqual(cfg.terrain.mesh_type, 'plane')
        self.assertFalse(cfg.terrain.curriculum)
        self.assertTrue(cfg.hardware_deployment_blocked)

    def test_gaussian_export_does_not_add_tanh_bound(self):
        with contextlib.redirect_stdout(io.StringIO()):
            model = HIMActorCritic(270, 238, 45, 12, actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16]).eval()
        history = torch.zeros((8, 270))
        with torch.no_grad():
            model.actor[-1].weight.zero_()
            model.actor[-1].bias.fill_(8.)
            with tempfile.TemporaryDirectory() as directory:
                exported = torch.jit.load(EXPORT.export_him_policy(model, directory))
                torch.testing.assert_close(exported(history), model.act_inference(history), rtol=0, atol=0)
                torch.testing.assert_close(exported(history), torch.full((8, 12), 8.), rtol=0, atol=0)

    def test_export_marks_gaussian_baseline_blocked_and_names_correct_task(self):
        env = observation_fixture()
        env.dof_names = [f'{leg}_{joint}_joint' for leg in ('FL', 'FR', 'RL', 'RR') for joint in ('hip', 'thigh', 'calf')]
        env.num_obs, env.history_length, env.dt = 270, 6, .02
        env.p_gains, env.d_gains, env.torque_limits = torch.ones(12)*20, torch.ones(12)*.5, torch.ones(12)*23.7
        env.cfg = NS(training_profile='go2_him_upstream_flat_v1', training_task='go2_him_upstream_flat',
                     action_distribution='gaussian', hardware_deployment_blocked=True,
                     control=NS(action_scale=.25, hip_reduction=1.),
                     normalization=NS(clip_actions=100., clip_observations=100.),
                     commands=NS(ranges=NS(lin_vel_x=[-1, 1], lin_vel_y=[-1, 1], ang_vel_yaw=[-3.14, 3.14])))
        with tempfile.TemporaryDirectory() as directory:
            result = yaml.safe_load(Path(DEPLOY.export_deploy_config(env, directory, 'go2')).read_text())
        self.assertEqual(result['policy']['action_distribution'], 'gaussian')
        self.assertNotIn('action_bound', result['policy'])
        self.assertTrue(result['hardware_deployment_blocked'])
        self.assertEqual(result['training_source']['task'], 'go2_him_upstream_flat')
