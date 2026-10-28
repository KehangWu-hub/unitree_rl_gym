"""Start a fresh, recorded HIM run; optionally detach it from the terminal."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--background',action='store_true')
    profiles = {'direct': 'go2_him_retrain', 'upstream-flat': 'go2_him_upstream_flat',
                'candidate-flat': 'go2_him_candidate_flat', 'candidate-rough': 'go2_him_candidate_rough'}
    parser.add_argument('--profile',choices=tuple(profiles),default='direct')
    own,remaining=parser.parse_known_args()
    directory=own.run_dir.resolve()
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'manifest.json').exists():
        raise ValueError('Refusing to overwrite a recorded training run')
    if own.background:
        command=[sys.executable,str(Path(__file__).resolve()),'--run-dir',str(directory),'--profile',own.profile,*remaining]
        with (directory/'console.log').open('x') as stream:
            process=subprocess.Popen(command,cwd=str(ROOT),stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(directory/'launch.json',{'pid':process.pid,'command':command,'run_dir':str(directory)})
        print(json.dumps({'pid':process.pid,'run_dir':str(directory)},ensure_ascii=False),flush=True)
        return

    import isaacgym
    import numpy as np
    import torch
    import yaml
    from legged_gym.envs import task_registry
    from legged_gym.utils import get_args
    from legged_gym.utils.helpers import class_to_dict
    from legged_gym.envs.go2_him.export import export_him_policy
    from legged_gym.utils.deploy import export_deploy_config
    from third_party.himloco_rsl_rl.rsl_rl.runners import HIMOnPolicyRunner
    sys.argv=[sys.argv[0]]+remaining
    args=get_args()
    if args.resume:
        raise ValueError('This entry is for fresh random initialization only')
    args.task=profiles[own.profile]
    torch.set_num_threads(1)
    cfg,training=task_registry.get_cfgs(args.task)
    cfg.seed=args.seed if args.seed is not None else training.seed
    training.seed=cfg.seed
    if args.max_iterations is not None:training.runner.max_iterations=args.max_iterations
    if args.num_envs is not None:cfg.env.num_envs=args.num_envs
    if own.profile=='direct':
        assert cfg.normalization.clip_actions==training.policy.action_bound==4.
    else:
        assert training.runner.policy_class_name=='HIMActorCritic'
        assert training.runner.num_steps_per_env==100
        assert training.algorithm.schedule=='adaptive'
    assert abs(cfg.sim.dt*cfg.control.decimation-.02)<1e-8
    assert not training.runner.resume
    env,actual_cfg=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    config=class_to_dict(training)
    def builtin(value):
        return json.loads(json.dumps(value,default=lambda item:item.item() if isinstance(item,np.generic) else item.tolist()))
    (directory/'env.yaml').write_text(yaml.safe_dump(builtin(class_to_dict(actual_cfg)),sort_keys=False))
    (directory/'train.yaml').write_text(yaml.safe_dump(builtin(config),sort_keys=False))
    if hasattr(env, 'terrain') and actual_cfg.terrain.mesh_type in ('heightfield', 'trimesh'):
        terrain = env.terrain
        np.savez_compressed(directory/'terrain.npz', heights=terrain.height_field_raw,
                            origins=terrain.env_origins)
        write_json(directory/'terrain.json', {
            'mesh_type':actual_cfg.terrain.mesh_type,
            'curriculum':actual_cfg.terrain.curriculum,
            'proportions':list(actual_cfg.terrain.terrain_proportions),
            'initial_levels':torch.bincount(env.terrain_levels).cpu().tolist(),
            'shape':list(terrain.height_field_raw.shape),
            'height_range_m':[float(terrain.height_field_raw.min()*actual_cfg.terrain.vertical_scale),
                              float(terrain.height_field_raw.max()*actual_cfg.terrain.vertical_scale)],
            'height_sha256':hashlib.sha256(terrain.height_field_raw.tobytes()).hexdigest(),
            'generator_sha256':hashlib.sha256((ROOT/'legged_gym/utils/terrain.py').read_bytes()).hexdigest(),
        })
    sources={}
    for base in ('legged_gym/envs/go2_him','third_party/himloco_rsl_rl/rsl_rl'):
        for path in sorted((ROOT/base).rglob('*.py')):
            relative=path.relative_to(ROOT)
            target=directory/'sources'/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            sources[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('legged_gym/scripts/train_him_from_scratch.py','legged_gym/scripts/evaluate_him_candidate.py',
                 'deploy/deploy_mujoco/evaluate_him_checkpoint.py',
                 'legged_gym/envs/__init__.py',
                 'legged_gym/utils/helpers.py','legged_gym/utils/deploy.py','legged_gym/utils/terrain.py',
                 'resources/robots/go2/urdf/go2.urdf'):
        path=ROOT/name;target=directory/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (directory/'source_manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
    (directory/'workspace.patch').write_text(subprocess.check_output(
        ['git','diff','HEAD'],cwd=str(ROOT),encoding='utf-8'),encoding='utf-8')
    runner=HIMOnPolicyRunner(env,config,str(directory),device=args.rl_device)
    initial_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in runner.alg.actor_critic.parameters())).hexdigest()
    manifest={'initialization':'fresh_random','checkpoint_loaded':None,'seed':actual_cfg.seed,
        'task':args.task,'num_envs':env.num_envs,'max_iterations':training.runner.max_iterations,
        'steps_per_env':training.runner.num_steps_per_env,'save_interval':training.runner.save_interval,
        'policy_class':training.runner.policy_class_name,
        'action_distribution':getattr(actual_cfg,'action_distribution','scaled_tanh_normal'),
        'action_bound':getattr(runner.alg.actor_critic,'action_bound',None),
        'training_action_clip':actual_cfg.normalization.clip_actions,
        'training_profile':actual_cfg.training_profile,'joint_target_mode':'direct',
        'software_motion_limits':False,'target_height_m':.30,'initial_parameter_sha256':initial_hash,
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip(),
        'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name(0),
        'pid':os.getpid(),'started_at':datetime.now().isoformat()}
    write_json(directory/'manifest.json',manifest)
    runner.save(str(directory/'model_0.pt'))
    exported=directory/'exported'
    export_deploy_config(env,str(exported/'params'),'go2')
    original_log=runner.log
    def log(locs,*args,**kwargs):
        losses={key:float(locs[name]) for key,name in {
            'value':'mean_value_loss','surrogate':'mean_surrogate_loss',
            'estimation':'mean_estimation_loss','swap':'mean_swap_loss'}.items()}
        if not all(math.isfinite(x) for x in losses.values()):
            raise RuntimeError('Non-finite training loss; refusing to continue')
        original_log(locs,*args,**kwargs)
        iteration=locs['it']+1
        write_json(directory/'progress.json',{'state':'training','iteration':iteration,
            'max_iterations':training.runner.max_iterations,'losses':losses,
            'transitions':iteration*env.num_envs*training.runner.num_steps_per_env,
            'mean_episode_reward':float(np.mean(locs['rewbuffer'])) if locs['rewbuffer'] else None,
            'mean_episode_length':float(np.mean(locs['lenbuffer'])) if locs['lenbuffer'] else None,
            'iteration_seconds':locs['collection_time']+locs['learn_time'],
            'std':float(runner.alg.actor_critic.action_std.mean().detach()),'pid':os.getpid()})
    runner.log=log
    try:
        runner.learn(training.runner.max_iterations,init_at_random_ep_len=True)
        export_him_policy(runner.alg.actor_critic,exported/'policies')
        progress=json.loads((directory/'progress.json').read_text());progress['state']='training_finished_pending_evaluation'
        write_json(directory/'progress.json',progress)
    except BaseException as error:
        write_json(directory/'failure.json',{'error':repr(error),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
