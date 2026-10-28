"""HIM temporal/supervision/export contracts, runnable without Isaac Gym."""

import ast
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

import torch
import yaml

from third_party.himloco_rsl_rl.rsl_rl.modules import HIMActorCritic
from third_party.himloco_rsl_rl.rsl_rl.runners import HIMOnPolicyRunner


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / "legged_gym/envs/go2_him/go2_him_env.py"


def file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPORT = file_module("him_export_test", ROOT / "legged_gym/envs/go2_him/export.py")
DEPLOY = file_module("him_deploy_test", ROOT / "legged_gym/utils/deploy.py")


def env_method(name):
    tree = ast.parse(ENV_FILE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Go2HIMRobot")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    # Reset tests use identity quaternions; test the selected state/history updates.
    scope = {"torch": torch, "quat_rotate_inverse": lambda q, vector: vector.clone()}
    exec(compile(module, str(ENV_FILE), "exec"), scope)
    return scope[name]


def observation_fixture():
    root = torch.zeros(2, 13)
    root[:, 2], root[:, 6] = .42, 1.
    root[:, 7:10] = torch.tensor([.1, .2, .3])
    env = NS(num_actions=12, num_one_step_obs=45, num_one_step_privileged_obs=238,
        obs_buf=torch.full((2, 270), -99.), privileged_obs_buf=torch.full((2, 238), -99.),
        commands=torch.tensor([[.5, .4, .2], [-.5, -.4, -.2]]),
        commands_scale=torch.tensor([2., 2., .25]), base_ang_vel=torch.zeros(2, 3),
        base_lin_vel=torch.ones(2, 3) * 99, projected_gravity=torch.zeros(2, 3),
        dof_pos=torch.zeros(2, 12), default_dof_pos=torch.zeros(1, 12),
        dof_vel=torch.zeros(2, 12), actions=torch.ones(2, 12),
        root_states=root, base_quat=root[:, 3:7].clone(),
        gravity_vec=torch.tensor([[0., 0., -1.]]).repeat(2, 1),
        disturbance=torch.zeros(2, 1, 3), add_noise=False,
        noise_scale_vec=torch.ones(232), measured_heights=torch.zeros(2, 187),
        cfg=NS(terrain=NS(measure_heights=True)),
        obs_scales=NS(ang_vel=.25, lin_vel=2., dof_pos=1., dof_vel=.05, height_measurements=5.))
    return env


class HIMTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def network(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return HIMActorCritic(270, 238, 45, 12, actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16])

    def test_reset_removes_only_selected_episode_history_and_stale_velocity(self):
        env = observation_fixture()
        env_method("_reset_observation_history")(env, torch.tensor([1]))
        self.assertTrue(torch.all(env.obs_buf[0] == -99))
        self.assertEqual(int(torch.count_nonzero(env.obs_buf[1])), 0)
        self.assertEqual(int(torch.count_nonzero(env.privileged_obs_buf[1])), 0)
        self.assertEqual(int(torch.count_nonzero(env.actions[1])), 0)
        torch.testing.assert_close(env.base_lin_vel[1], torch.tensor([.1, .2, .3]))
        env_method("compute_observations")(env)
        torch.testing.assert_close(env.obs_buf[1, :3], torch.tensor([-1., -.8, -.05]))
        self.assertEqual(int(torch.count_nonzero(env.obs_buf[1, 45:])), 0)
        old = env.obs_buf[:, :45].clone()
        env.commands[:, 0] = .25
        env_method("compute_observations")(env)
        torch.testing.assert_close(env.obs_buf[:, 45:90], old)
        torch.testing.assert_close(env.obs_buf[:, 0], torch.full((2,), .5))

    def test_terminal_supervision_uses_pre_reset_velocity_without_noise(self):
        env = observation_fixture()
        env.base_lin_vel[:] = torch.tensor([.3, -.2, .1])
        terminal = env_method("compute_termination_observations")(env, torch.tensor([1]))
        torch.testing.assert_close(terminal[0, 45:48], torch.tensor([.6, -.4, .2]))
        env_method("_reset_observation_history")(env, torch.tensor([1]))
        torch.testing.assert_close(terminal[0, 45:48], torch.tensor([.6, -.4, .2]))
        env_method("compute_observations")(env)
        first = env.privileged_obs_buf.clone()
        env_method("compute_observations")(env)
        torch.testing.assert_close(env.privileged_obs_buf, first, rtol=0, atol=0)

    def test_export_matches_trained_network_and_retains_encoder(self):
        network = self.network().eval()
        history = torch.randn(16, 270)
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            path = EXPORT.export_him_policy(network, directory)
            loaded = torch.jit.load(path)
            torch.testing.assert_close(loaded(history), network.act_inference(history), rtol=0, atol=0)
            self.assertEqual(tuple(dict(loaded.named_parameters())["estimator.0.weight"].shape), (128, 270))
            with self.assertRaises((RuntimeError, torch.jit.Error)):
                loaded(torch.zeros(1, 45))

    def test_checkpoint_restores_both_optimizers_iteration_and_learning_rates(self):
        class EmptyEnv:
            num_envs, num_obs, num_privileged_obs, num_one_step_obs, num_actions = 4, 270, 238, 45, 12
            def reset(self):
                return torch.zeros(4, 270), torch.zeros(4, 238)
        config = dict(runner=dict(policy_class_name="HIMActorCritic", algorithm_class_name="HIMPPO",
            num_steps_per_env=4, save_interval=1), algorithm={},
            policy=dict(actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16]))
        with contextlib.redirect_stdout(io.StringIO()):
            runner = HIMOnPolicyRunner(EmptyEnv(), config, None)
            restored = HIMOnPolicyRunner(EmptyEnv(), config, None)
        network = runner.alg.actor_critic
        network.estimator.update(torch.randn(8, 270), torch.randn(8, 238))
        runner.alg.optimizer.zero_grad()
        network.actor(torch.randn(8, 64)).square().mean().backward()
        runner.alg.optimizer.step()
        runner.current_learning_iteration = 7
        runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]['lr'] = .0002
        history = torch.randn(4, 270)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model_7.pt")
            runner.save(path)
            restored.load(path)
        self.assertEqual(restored.current_learning_iteration, 7)
        self.assertEqual(restored.alg.learning_rate, .0002)
        self.assertTrue(restored.alg.optimizer.state)
        self.assertTrue(restored.alg.actor_critic.estimator.optimizer.state)
        torch.testing.assert_close(network.act_inference(history), restored.alg.actor_critic.act_inference(history), rtol=0, atol=0)

    def test_exported_contract_has_temporal_layout_and_scaled_velocity(self):
        names = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]
        env = observation_fixture()
        env.dof_names, env.num_obs, env.history_length = names, 270, 6
        env.dt, env.commands_scale = .02, torch.tensor([2., 2., .25])
        env.p_gains, env.d_gains, env.torque_limits = torch.ones(12)*20, torch.ones(12)*.5, torch.ones(12)*23.7
        env.cfg.control = NS(action_scale=.25, hip_reduction=1.)
        env.cfg.normalization = NS(clip_actions=100., clip_observations=100.)
        env.cfg.commands = NS(ranges=NS(lin_vel_x=[-1., 1.], lin_vel_y=[-1., 1.], ang_vel_yaw=[-3.14, 3.14]))
        with tempfile.TemporaryDirectory() as directory:
            cfg = yaml.safe_load(Path(DEPLOY.export_deploy_config(env, directory, "go2")).read_text())
        self.assertEqual(cfg['format_version'], 2)
        self.assertEqual(cfg['policy']['num_observations'], 270)
        self.assertEqual(cfg['policy']['history_length'], 6)
        self.assertEqual(cfg['policy']['estimated_velocity_scale'], 2.)
        self.assertEqual(cfg['observations'][0]['name'], 'velocity_commands')
        self.assertEqual(sum(term['size'] for term in cfg['observations']), 45)
        self.assertEqual(cfg['policy']['policy_to_sdk'], [3,4,5,0,1,2,9,10,11,6,7,8])
        env.cfg.training_profile = 'go2_him_direct_targets_v3'
        env.cfg.normalization.clip_actions = 4.
        with tempfile.TemporaryDirectory() as directory:
            direct = yaml.safe_load(Path(DEPLOY.export_deploy_config(env, directory, 'go2')).read_text())
        self.assertFalse(direct['software_motion_limits'])
        self.assertEqual(direct['control']['joint_target_mode'], 'direct')
        self.assertNotIn('policy_pd_limit_nm', direct['control'])
        self.assertNotIn('action_step_limit', direct['control'])


if __name__ == "__main__":
    unittest.main()
