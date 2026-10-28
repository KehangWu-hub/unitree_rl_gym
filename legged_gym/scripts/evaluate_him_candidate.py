"""50-second forward/reverse/stop trials for fresh HIM candidates; simulation only."""
import argparse,json,sys,hashlib
from pathlib import Path
import isaacgym
import numpy as np
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.envs.go2_him.acceptance import assess_motion

ROOT=Path(LEGGED_GYM_ROOT_DIR)


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--policy-path',type=Path,required=True)
    parser.add_argument('--output-path',type=Path,required=True)
    parser.add_argument('--command-profile',choices=('full','keyboard'),default='full')
    own,remaining=parser.parse_known_args();sys.argv=[sys.argv[0]]+remaining
    args=get_args();args.task='go2_him_retrain';args.num_envs=args.num_envs or 64
    if args.num_envs%8:raise ValueError('Environment count must be a multiple of eight')
    cfg,_=task_registry.get_cfgs(args.task)
    cfg.env.num_envs=args.num_envs;cfg.env.episode_length_s=120.;cfg.noise.add_noise=False
    cfg.commands.resampling_time=1e6
    for key in dir(cfg.domain_rand):
        if key.startswith('randomize_') or key in ('disturbance','push_robots','delay'):
            setattr(cfg.domain_rand,key,False)
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    env.reset()
    # Establish a standing initial condition before scoring locomotion. Otherwise
    # the simulator's uninitialized zero-angle joints create a spurious reset.
    nominal_kp,nominal_kd=env.p_gains.clone(),env.d_gains.clone()
    env.p_gains*=2.;env.d_gains*=2.
    for _ in range(150):
        env.step(torch.zeros_like(env.actions))
    env.p_gains[:]=nominal_kp;env.d_gains[:]=nominal_kd
    policy=torch.jit.load(str(own.policy_path),map_location=env.device).eval()
    commands=torch.tensor([[0.,0,0],[.3,0,0],[-.3,0,0],[0,.3,0],[0,-.3,0],[.3,0,.3],[.3,0,-.3],[0,0,.3]],device=env.device).repeat_interleave(args.num_envs//8,dim=0)
    env.commands[:,:3]=commands
    env._reset_observation_history(torch.arange(args.num_envs,device=env.device))
    env.compute_observations();obs=env.get_observations()
    resets=torch.zeros(args.num_envs,device=env.device);action_max=torch.zeros(args.num_envs,device=env.device)
    samples=[]
    with torch.inference_mode():
        for step in range(2500):
            time=step*.02
            reversed_commands=-commands.clone()
            if own.command_profile=='keyboard':
                # Q/E reverse yaw while retaining forward motion, exactly as
                # the single-key interface; backward+yaw is a separate ability.
                reversed_commands[(commands[:,0]>0)&(commands[:,2]!=0),0]=commands[(commands[:,0]>0)&(commands[:,2]!=0),0]
            requested=commands if time<20. else reversed_commands if time<40. else torch.zeros_like(commands)
            env.commands[:,:3]=requested
            action=policy(obs);action_max=torch.maximum(action_max,action.abs().max(dim=1).values)
            obs,_,_,done,*_=env.step(action);resets+=done
            velocity=torch.cat((env.base_lin_vel[:,:2],env.base_ang_vel[:,2:3]),dim=1)
            tilt=torch.acos(torch.clamp(-env.projected_gravity[:,2],-1.,1.))
            row=torch.cat((torch.full((args.num_envs,1),time,device=env.device),requested,velocity,env.root_states[:,2:3],tilt[:,None]),dim=1)
            samples.append(row.cpu().numpy())
    data=np.asarray(samples);names=('zero','w','s','a','d','q','e','yaw')
    reports=[];group=args.num_envs//8
    for index,name in enumerate(names):
        section=slice(index*group,(index+1)*group)
        result=assess_motion(data[:,section,:],int(resets[section].sum()),float(action_max[section].max()))
        result['case']=name;reports.append(result)
    report={'simulation_only':True,'command_profile':own.command_profile,
            'policy_sha256':hashlib.sha256(own.policy_path.read_bytes()).hexdigest(),
            'passed':all(r['passed'] for r in reports),'cases':reports,
            'policy_path':str(own.policy_path),'duration_seconds':50.,'envs':args.num_envs}
    own.output_path.parent.mkdir(parents=True,exist_ok=True)
    own.output_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(own.output_path.with_suffix('.npz'),samples=data)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
