import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

REWARDS_PATH = Path(__file__).parents[1] / "legged_gym" / "envs" / "go2" / "rewards.py"
SPEC = importlib.util.spec_from_file_location("go2_reward_terms", REWARDS_PATH)
REWARDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REWARDS)


class Go2TrainingContractTest(unittest.TestCase):
    def test_translated_joint_position_penalty(self):
        env = SimpleNamespace(
            commands=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            base_lin_vel=torch.zeros(2, 3),
            dof_pos=torch.ones(2, 12),
            default_dof_pos=torch.zeros(1, 12),
        )
        reward = REWARDS.joint_position_penalty(
            env.dof_pos, env.default_dof_pos, env.commands, env.base_lin_vel
        )
        expected = torch.sqrt(torch.tensor(12.0))
        torch.testing.assert_close(reward, torch.tensor([5.0 * expected, expected]))

    def test_translated_energy_penalty(self):
        env = SimpleNamespace(
            dof_vel=torch.tensor([[1.0, -2.0, 3.0]]),
            torques=torch.tensor([[-4.0, 5.0, -6.0]]),
        )
        torch.testing.assert_close(REWARDS.energy(env.dof_vel, env.torques), torch.tensor([32.0]))


if __name__ == "__main__":
    unittest.main()
