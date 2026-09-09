from legged_gym.envs.base.legged_robot import LeggedRobot
import torch


class Go2Robot(LeggedRobot):
    """使用真机可获取观测量的Go2环境。"""

    def compute_observations(self):
        """构造可部署的Actor观测和仅供仿真训练使用的Critic观测。"""
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

        # 遵循官方非对称结构：Critic额外获得仅仿真器可提供的机身速度和关节力矩。
        # 该60维缓冲区不会随45维Actor策略导出。
        self.privileged_obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.torques * 0.01,
                self.actions,
            ),
            dim=-1,
        )

        if self.add_noise:
            actor_obs = actor_obs + (2 * torch.rand_like(actor_obs) - 1) * self.noise_scale_vec

        self.obs_buf = actor_obs

    def _get_noise_scale_vec(self, cfg):
        """返回与Go2的45维观测逐项对齐的噪声缩放向量。"""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.0  # 速度指令
        noise_vec[9 : 9 + self.num_actions] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[9 + self.num_actions : 9 + 2 * self.num_actions] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = 0.0  # 上一次动作

        return noise_vec
