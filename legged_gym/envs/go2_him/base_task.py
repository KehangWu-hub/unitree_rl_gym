"""Adapt the shared Isaac Gym task lifecycle to HIM's seven-value step."""

import torch

from legged_gym.envs.base.base_task import BaseTask as SharedBaseTask


class BaseTask(SharedBaseTask):
    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        result = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device))
        return result[0], result[1]
