"""Measure a complete HIM TorchScript policy in deterministic, flat Isaac Gym.

Run separately from training: simultaneous PhysX instances can exhaust GPU memory.
"""
import argparse
import json
from pathlib import Path
import sys

import isaacgym
from isaacgym import gymtorch
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from deploy.common.him_qualification import sha256

ROOT = Path(LEGGED_GYM_ROOT_DIR)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--policy-path', type=Path, default=ROOT/'deploy/pre_train/go2_him/policy.pt')
    parser.add_argument('--output-path', type=Path, default=ROOT/'logs/go2_him/phase2_validation/isaac.json')
    own, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    args.task = 'go2_him_retrain' if args.task == 'go2_him_retrain' else 'go2_him'
    args.num_envs = args.num_envs or 64
    if args.num_envs < 8 or args.num_envs % 8:
        raise ValueError('Use a positive multiple of eight environments')
    cfg, _ = task_registry.get_cfgs(args.task)
    cfg.env.num_envs = args.num_envs
    cfg.env.test = False
    cfg.env.episode_length_s = 30.
    cfg.terrain.mesh_type = 'plane'
    cfg.terrain.curriculum = False
    cfg.noise.add_noise = False
    cfg.commands.heading_command = False
    cfg.commands.resampling_time = 1e6
    for field in dir(cfg.domain_rand):
        if field.startswith('randomize_') or field in ('push_robots', 'disturbance', 'delay'):
            setattr(cfg.domain_rand, field, False)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    env.reset()
    saved_kp,saved_kd = env.p_gains.clone(),env.d_gains.clone()
    env.p_gains *= 2.;env.d_gains *= 2.
    for _ in range(150):env.step(torch.zeros_like(env.actions))
    env.p_gains[:] = saved_kp;env.d_gains[:] = saved_kd
    policy = torch.jit.load(str(own.policy_path), map_location=env.device).eval()
    cases = [[0,0,0], [.3,0,0], [-.3,0,0], [0,.3,0], [0,-.3,0], [.3,0,.3], [.3,0,-.3], [0,0,0]]
    commands = torch.tensor(cases, device=env.device).repeat_interleave(args.num_envs//8, dim=0)
    env.commands[:,:3] = commands
    env._reset_observation_history(torch.arange(args.num_envs,device=env.device))
    env.compute_observations()
    obs = env.get_observations()
    values = []
    resets = torch.zeros(args.num_envs, device=env.device)
    with torch.inference_mode():
        for step in range(650):
            env.commands[:,:3] = commands if step < 500 else 0
            obs, _, _, done, *_ = env.step(policy(obs))
            resets += done
            if 100 <= step < 500:
                values.append(torch.cat((env.base_lin_vel[:,:2], env.base_ang_vel[:,2:3]), dim=-1).cpu().numpy())
    velocity = np.asarray(values).mean(axis=0).reshape(8, args.num_envs//8, 3).mean(axis=1)
    props = env.gym.get_actor_rigid_shape_properties(env.envs[0], env.actor_handles[0])
    friction = {name:float(getattr(props[-1], name)) for name in ('friction', 'rolling_friction', 'torsion_friction')}
    report = {'simulation_only':True, 'finite':bool(np.isfinite(values).all()),
              'policy_sha256':sha256(own.policy_path), 'mean_velocity':velocity.tolist(),
              'commands':cases, 'labels':['zero','w','s','a','d','q','e','zero'],
              'contact_friction':friction, 'resets':resets.cpu().tolist(),
              'num_envs':args.num_envs, 'sample_window_seconds':[2.,10.], 'seed':cfg.seed,
              'environment_source_sha256':sha256(ROOT/'legged_gym/envs/go2_him/go2_him_env.py')}
    own.output_path.parent.mkdir(parents=True, exist_ok=True)
    own.output_path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    return 0 if report['finite'] and not resets.any() else 1


if __name__ == '__main__':
    raise SystemExit(main())
