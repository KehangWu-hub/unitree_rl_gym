"""Matched terrain candidates preserve the selected recipe and local clearance."""
import ast
from pathlib import Path
import unittest

import torch

from tests.test_go2_him_upstream_flat import config_scope

ROOT = Path(__file__).resolve().parents[1]


def candidate_scope():
    scope = config_scope()
    for name in ('retrain_config.py', 'candidate_config.py'):
        path = ROOT/'legged_gym/envs/go2_him'/name
        tree = ast.parse(path.read_text())
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, str(path), 'exec'), scope)
    return scope


class CandidateTest(unittest.TestCase):
    def test_selected_parameters_do_not_blindly_replace_v2_recipe(self):
        scope = candidate_scope()
        cfg, ppo = scope['Go2HIMCandidateRoughCfg'](), scope['Go2HIMCandidateRoughCfgPPO']()
        self.assertEqual(cfg.env.num_envs, 1024)
        self.assertEqual(cfg.sim.dt, .002)
        self.assertEqual(cfg.control.decimation, 10)
        self.assertEqual(ppo.runner.num_steps_per_env, 100)
        self.assertEqual(ppo.runner.max_iterations*cfg.env.num_envs*100, 131072000)
        self.assertEqual(ppo.runner.policy_class_name, 'HIMActorCritic')
        self.assertEqual(ppo.policy.init_noise_std, .6)
        self.assertFalse(hasattr(ppo.policy, 'action_bound'))
        self.assertEqual(ppo.algorithm.learning_rate, .0003)
        self.assertEqual(ppo.algorithm.schedule, 'adaptive')
        self.assertEqual(ppo.algorithm.entropy_coef, .005)
        self.assertIs(scope['Go2HIMCandidateFlatCfg'].rewards, scope['Go2HIMRetrainCfg'].rewards)
        self.assertIs(scope['Go2HIMCandidateFlatCfg'].commands, scope['Go2HIMRetrainCfg'].commands)
        self.assertTrue(cfg.hardware_deployment_blocked)

    def test_rough_variant_only_changes_terrain_and_run_labels(self):
        scope = candidate_scope()
        flat, rough = scope['Go2HIMCandidateFlatCfg'](), scope['Go2HIMCandidateRoughCfg']()
        self.assertEqual(flat.terrain.mesh_type, 'plane')
        self.assertEqual(rough.terrain.mesh_type, 'trimesh')
        self.assertTrue(rough.terrain.curriculum)
        self.assertEqual(rough.terrain.max_init_terrain_level, 1)
        self.assertEqual(rough.terrain.terrain_proportions, [.1, .1, .35, .25, .2])
        self.assertEqual(flat.control.action_scale, rough.control.action_scale)
        self.assertEqual(flat.rewards.scales.feet_slip, rough.rewards.scales.feet_slip)

    def test_low_clearance_termination_uses_local_ground_on_stairs(self):
        class Base:
            def check_termination(self):
                self.reset_buf = torch.zeros(3, dtype=torch.bool)
        path = ROOT/'legged_gym/envs/go2_him/retrain_env.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'check_termination']
        scope = {'Go2HIMRobot': Base, 'torch': torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), 'exec'), scope)
        env = scope['Go2HIMRetrainRobot']()
        env.projected_gravity = torch.tensor([[0., 0., -1.]]).repeat(3, 1)
        env.root_states = torch.zeros(3, 13)
        env.root_states[:, 2] = torch.tensor([.10, 1.10, .30])
        env._get_base_heights = lambda: torch.tensor([.30, .10, .30])
        env.dof_vel_limits = torch.ones(12)*25
        env.dof_vel = torch.zeros(3, 12)
        env.dof_pos = torch.zeros(3, 12)
        env.dof_pos_limits = torch.tensor([[-1., 1.]]).repeat(12, 1)
        env.check_termination()
        self.assertEqual(env.reset_buf.tolist(), [False, True, False])
