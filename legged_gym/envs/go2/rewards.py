import torch


def energy(dof_vel, torques):
    """Joint mechanical-power magnitude used by the official Go2 reward."""
    return torch.sum(torch.abs(dof_vel) * torch.abs(torques), dim=1)


def joint_position_penalty(dof_pos, default_dof_pos, commands, base_lin_vel):
    """Default-pose deviation with the official fivefold idle multiplier."""
    command_speed = torch.linalg.norm(commands[:, :3], dim=1)
    body_speed = torch.linalg.norm(base_lin_vel[:, :2], dim=1)
    deviation = torch.linalg.norm(dof_pos - default_dof_pos, dim=1)
    moving = torch.logical_or(command_speed > 0.0, body_speed > 0.3)
    return torch.where(moving, deviation, 5.0 * deviation)
