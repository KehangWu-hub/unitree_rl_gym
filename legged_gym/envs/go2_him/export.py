"""Export the complete HIM inference graph, including its history encoder."""

import copy
from pathlib import Path

import torch
import torch.nn.functional as F


class HIMPolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor).cpu()
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder).cpu()
        self.num_one_step_obs = int(actor_critic.num_one_step_obs)

    def forward(self, obs_history):
        parts = self.estimator(obs_history)
        velocity = parts[..., :3]
        latent = F.normalize(parts[..., 3:], dim=-1, p=2.0)
        return self.actor(torch.cat((obs_history[..., :self.num_one_step_obs], velocity, latent), dim=-1))


def export_him_policy(actor_critic, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "policy.pt"
    model = HIMPolicyExporter(actor_critic).eval()
    torch.jit.script(model).save(str(destination))
    return str(destination)
