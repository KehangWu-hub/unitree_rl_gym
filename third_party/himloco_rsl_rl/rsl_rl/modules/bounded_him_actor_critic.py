"""HIM with a bounded action distribution and its exact PPO log density."""
import math
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from .him_actor_critic import HIMActorCritic


class BoundedHIMActorCritic(HIMActorCritic):
    def __init__(self, *args, action_bound=4., init_noise_std=.6, **kwargs):
        super().__init__(*args, init_noise_std=init_noise_std, **kwargs)
        if not math.isfinite(action_bound) or action_bound <= 0 or init_noise_std <= 0:
            raise ValueError('Action bound and initial standard deviation must be positive')
        self.action_bound = float(action_bound)
        initial = self.std.detach().clone()
        del self.std
        self.log_std = nn.Parameter(initial.log())

    def update_distribution(self, obs_history):
        velocity, latent = self.estimator(obs_history)
        mean = self.actor(torch.cat((obs_history[:, :self.num_one_step_obs], velocity, latent), dim=-1))
        std = self.log_std.clamp(-4., .5).exp().expand_as(mean)
        self.distribution = Normal(mean, std)

    def _squash(self, raw):
        return self.action_bound * torch.tanh(raw / self.action_bound)

    def _log_jacobian(self, raw):
        value = raw / self.action_bound
        return 2. * (math.log(2.) - value - F.softplus(-2. * value))

    def act(self, obs_history=None, **kwargs):
        self.update_distribution(obs_history)
        return self._squash(self.distribution.sample())

    def get_actions_log_prob(self, actions):
        value = (actions / self.action_bound).clamp(-1. + 1e-6, 1. - 1e-6)
        raw = self.action_bound * .5 * (torch.log1p(value) - torch.log1p(-value))
        return (self.distribution.log_prob(raw) - self._log_jacobian(raw)).sum(dim=-1)

    @property
    def entropy(self):
        # Monte Carlo entropy of the transformed distribution, including its Jacobian.
        raw = self.distribution.rsample()
        return self.distribution.entropy().sum(dim=-1) + self._log_jacobian(raw).sum(dim=-1)

    def act_inference(self, obs_history, observations=None):
        return self._squash(super().act_inference(obs_history, observations))
