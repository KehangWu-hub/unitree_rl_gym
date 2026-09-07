from legged_gym.envs.base.legged_robot import LeggedRobot
import torch


class Go2Robot(LeggedRobot):
    """Go2 environment with observations available on the physical robot."""

    def compute_observations(self):
        """Build deployable actor observations and simulator-only critic observations."""
        actor_obs = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

        # Keep the original 48-dimensional observation for the asymmetric
        # critic.  This buffer is never exported with the actor policy.
        self.privileged_obs_buf = torch.cat(
            (self.base_lin_vel * self.obs_scales.lin_vel, actor_obs), dim=-1
        )

        if self.add_noise:
            actor_obs = actor_obs + (2 * torch.rand_like(actor_obs) - 1) * self.noise_scale_vec

        self.obs_buf = actor_obs

    def _get_noise_scale_vec(self, cfg):
        """Return noise scales aligned with the Go2 45-dimensional observation."""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.0  # velocity commands
        noise_vec[9 : 9 + self.num_actions] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[9 + self.num_actions : 9 + 2 * self.num_actions] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = 0.0  # previous actions

        return noise_vec
