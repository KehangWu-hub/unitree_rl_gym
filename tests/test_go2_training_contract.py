import unittest
import ast
from pathlib import Path
from types import SimpleNamespace

import torch

LEGGED_ROBOT_PATH = (
    Path(__file__).parents[1] / "legged_gym" / "envs" / "base" / "legged_robot.py"
)


def load_reward_method(name):
    """在不导入Isaac Gym的情况下加载一个真实的基类奖励方法。"""
    tree = ast.parse(LEGGED_ROBOT_PATH.read_text())
    legged_robot = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LeggedRobot"
    )
    method = next(
        node
        for node in legged_robot.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, str(LEGGED_ROBOT_PATH), "exec"), namespace)
    return namespace[name]


REWARD_JOINT_POS = load_reward_method("_reward_joint_pos")
REWARD_ENERGY = load_reward_method("_reward_energy")


class Go2TrainingContractTest(unittest.TestCase):
    def test_translated_joint_position_penalty(self):
        env = SimpleNamespace(
            commands=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            base_lin_vel=torch.zeros(2, 3),
            dof_pos=torch.ones(2, 12),
            default_dof_pos=torch.zeros(1, 12),
        )
        reward = REWARD_JOINT_POS(env)
        expected = torch.sqrt(torch.tensor(12.0))
        torch.testing.assert_close(reward, torch.tensor([5.0 * expected, expected]))

    def test_translated_energy_penalty(self):
        env = SimpleNamespace(
            dof_vel=torch.tensor([[1.0, -2.0, 3.0]]),
            torques=torch.tensor([[-4.0, 5.0, -6.0]]),
        )
        torch.testing.assert_close(REWARD_ENERGY(env), torch.tensor([32.0]))


if __name__ == "__main__":
    unittest.main()
