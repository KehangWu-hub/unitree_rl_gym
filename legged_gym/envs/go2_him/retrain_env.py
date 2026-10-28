"""Go2 HIM training with direct joint targets and model actuator limits."""
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float
from .go2_him_env import Go2HIMRobot


class Go2HIMRetrainRobot(Go2HIMRobot):
    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        self.root_states[env_ids, 7:13] *= .2
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids.to(dtype=torch.int32)), len(env_ids))

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        # Equal coverage: zero, forward, backward, left, right, two walking turns, yaw.
        modes = torch.randint(0, 8, (len(env_ids),), device=self.device)
        self.commands[env_ids] = 0.
        x = torch_rand_float(.10, self.command_ranges['lin_vel_x'][1], (len(env_ids),1), device=self.device).squeeze(1)
        y = torch_rand_float(.10, self.command_ranges['lin_vel_y'][1], (len(env_ids),1), device=self.device).squeeze(1)
        yaw = torch_rand_float(.15, self.command_ranges['ang_vel_yaw'][1], (len(env_ids),1), device=self.device).squeeze(1)
        self.commands[env_ids,0] = x * ((modes==1) | (modes==5) | (modes==6)) - x * (modes==2)
        self.commands[env_ids,1] = y * (modes==3) - y * (modes==4)
        signs = torch.where(torch.rand(len(env_ids),device=self.device)>.5, 1., -1.)
        self.commands[env_ids,2] = yaw * (modes==5) - yaw * (modes==6) + yaw * signs * (modes==7)

    def _compute_torques(self, actions):
        # Match unrestricted deployment: no slew filter, target clipping, or
        # nominal PD projection. Only the physical actuator model clips torque.
        self.joint_pos_target = self.default_dof_pos + actions * self.cfg.control.action_scale
        kp, kd = self.p_gains * self.Kp_factors, self.d_gains * self.Kd_factors
        actual = (kp*(self.joint_pos_target-self.dof_pos)-kd*self.dof_vel) * self.motor_strength_factors
        return torch.clip(actual, -self.torque_limits, self.torque_limits)

    def check_termination(self):
        super().check_termination()
        tilt = torch.acos(torch.clamp(-self.projected_gravity[:,2],-1.,1.))
        velocity_limit = torch.minimum(self.dof_vel_limits,torch.full_like(self.dof_vel_limits,25.))
        # Use clearance above the local terrain, not world Z on hills/stairs.
        self.reset_buf |= (tilt > .35) | (self._get_base_heights() < .18)
        self.reset_buf |= (torch.abs(self.dof_vel)>velocity_limit).any(dim=1)
        outside = (self.dof_pos < self.dof_pos_limits[:,0]-.02) | (self.dof_pos > self.dof_pos_limits[:,1]+.02)
        self.reset_buf |= outside.any(dim=1)

    def _reward_upright_height(self):
        error = self._get_base_heights() - self.cfg.rewards.base_height_target
        return torch.exp(-torch.square(error)/.0025)

    def _reward_idle_motion(self):
        idle = torch.linalg.norm(self.commands[:,:3],dim=1)<.05
        motion = torch.sum(self.dof_vel.square(),dim=1) + 20*torch.sum(self.base_lin_vel[:,:2].square(),dim=1)
        motion += 5*torch.sum(self.base_ang_vel.square(),dim=1)
        return motion*idle

    def _reward_feet_slip(self):
        contact = self.contact_forces[:,self.feet_indices,2]>1.
        return torch.sum(torch.sum(self.feet_vel[:,:,:2].square(),dim=-1)*contact,dim=1)
