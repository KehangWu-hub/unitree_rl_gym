"""Wait for a recorded HIM run, then resume it to an absolute iteration target."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def active(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def overlay(obj, values):
    if isinstance(obj, dict):
        obj.clear()
        obj.update(copy.deepcopy(values))
        return
    for name, value in values.items():
        if isinstance(value, dict) and hasattr(obj, name):
            overlay(getattr(obj, name), value)
        else:
            setattr(obj, name, value)


def worker(source, out, target):
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
    torch.set_num_threads(1)
    parent = json.loads((source/'manifest.json').read_text())
    progress = json.loads((source/'progress.json').read_text())
    start = int(progress['iteration'])
    checkpoint = source/f'model_{start}.pt'
    if target <= start:
        raise ValueError('Target must exceed the source iteration')
    # Fail visibly rather than silently resume with a different training implementation.
    source_hashes = json.loads((source/'source_manifest.json').read_text())
    for name, digest in source_hashes.items():
        if name.startswith(('legged_gym/envs/go2_him/', 'third_party/himloco_rsl_rl/')) or name == 'legged_gym/utils/terrain.py':
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Training source changed: {name}')
    actual = yaml.safe_load((source/'env.yaml').read_text())
    train = yaml.safe_load((source/'train.yaml').read_text())
    train['runner']['max_iterations'] = target
    sys.argv = [sys.argv[0], '--task', parent['task'], '--num_envs', str(parent['num_envs']),
                '--seed', str(parent['seed']), '--headless']
    args = get_args()
    cfg, _ = task_registry.get_cfgs(parent['task'])
    overlay(cfg, actual)
    env, cfg = task_registry.make_env(parent['task'], args=args, env_cfg=cfg)
    if hasattr(env, 'terrain'):
        height_hash = hashlib.sha256(env.terrain.height_field_raw.tobytes()).hexdigest()
        expected = json.loads((source/'terrain.json').read_text())['height_sha256']
        if height_hash != expected:
            raise ValueError('Recreated terrain differs from the source run')
    runner = HIMOnPolicyRunner(env, train, str(out), device=args.rl_device)
    saved = torch.load(checkpoint, map_location=args.rl_device)
    infos = runner.load(str(checkpoint), load_optimizer=True)
    if runner.current_learning_iteration != start:
        raise ValueError('Loaded checkpoint iteration mismatch')
    if not all(torch.equal(tensor, saved['model_state_dict'][name])
               for name, tensor in runner.alg.actor_critic.state_dict().items()):
        raise ValueError('Loaded model parameters mismatch')
    if not runner.alg.optimizer.state or not runner.alg.actor_critic.estimator.optimizer.state:
        raise ValueError('Optimizer states were not restored')
    curriculum = (infos or {}).get('terrain_curriculum')
    if curriculum is not None:
        env.terrain_levels[:] = curriculum['levels'].to(env.device)
        env.terrain_types[:] = curriculum['types'].to(env.device)
        env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
        env.reset()
        curriculum_note = 'restored_from_checkpoint'
    else:
        curriculum_note = 'source_checkpoint_has_no_curriculum_state; original_low_level_initialization'
    (out/'env.yaml').write_text((source/'env.yaml').read_text())
    (out/'train.yaml').write_text(yaml.safe_dump(train, sort_keys=False))
    shutil.copy2(source/'terrain.json', out/'terrain.json')
    shutil.copy2(source/'terrain.npz', out/'terrain.npz')
    snapshot = out/'sources'
    hashes = {}
    for name in source_hashes:
        path = ROOT/name
        destination = snapshot/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(__file__, out/'continuation_script_snapshot.py')
    write(out/'source_manifest.json', hashes)
    manifest = dict(parent, initialization='checkpoint_resume', checkpoint_loaded=str(checkpoint),
                    start_iteration=start, max_iterations=target, additional_iterations=target-start,
                    pid=os.getpid(), started_at=datetime.now().isoformat(), source_run=str(source),
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    curriculum_restore=curriculum_note, environment_state_restored=False,
                    both_optimizers_restored=True, parameter_restore_exact=True)
    write(out/'manifest.json', manifest)
    export_deploy_config(env, str(out/'exported/params'), 'go2')
    original_save = runner.save
    def save(path, infos=None):
        extra = dict(infos or {})
        extra['terrain_curriculum'] = {'levels': env.terrain_levels.detach().cpu().clone(),
                                        'types': env.terrain_types.detach().cpu().clone()}
        original_save(path, infos=extra)
    runner.save = save
    runner.save(str(out/f'model_{start}.pt'))
    original_log = runner.log
    runner.tot_timesteps = start*env.num_envs*runner.num_steps_per_env
    def log(locs, *args, **kwargs):
        losses = {name: float(locs[key]) for name, key in {
            'value':'mean_value_loss', 'surrogate':'mean_surrogate_loss',
            'estimation':'mean_estimation_loss', 'swap':'mean_swap_loss'}.items()}
        if not all(math.isfinite(v) for v in losses.values()):
            raise RuntimeError('Non-finite resumed training losses')
        original_log(locs, *args, **kwargs)
        iteration = locs['it']+1
        write(out/'progress.json', {'state':'training', 'iteration':iteration, 'max_iterations':target,
            'start_iteration':start, 'transitions':iteration*env.num_envs*runner.num_steps_per_env,
            'losses':losses, 'iteration_seconds':locs['collection_time']+locs['learn_time'],
            'mean_episode_reward':float(np.mean(locs['rewbuffer'])) if locs['rewbuffer'] else None,
            'mean_episode_length':float(np.mean(locs['lenbuffer'])) if locs['lenbuffer'] else None,
            'pid':os.getpid()})
    runner.log = log
    write(out/'progress.json', {'state':'training', 'iteration':start, 'max_iterations':target,
                                'start_iteration':start, 'pid':os.getpid()})
    runner.learn(target-start, init_at_random_ep_len=True)
    export_him_policy(runner.alg.actor_critic, out/'exported/policies')
    progress = json.loads((out/'progress.json').read_text())
    progress['state'] = 'training_finished_pending_evaluation'
    write(out/'progress.json', progress)


def orchestrate(source, out, target):
    plan = {'state':'waiting_for_source', 'source_run':str(source), 'target_iteration':target,
            'pid':os.getpid(), 'hardware_actions':False}
    write(out/'auto_continuation.json', plan)
    while True:
        if (source/'failure.json').exists():
            raise RuntimeError('Source training failed; no silent fallback to a different checkpoint')
        if (source/'progress.json').exists():
            state = json.loads((source/'progress.json').read_text())
            if state.get('state') == 'training_finished_pending_evaluation' and not active(int(state['pid'])):
                break
        time.sleep(5)
    command = [sys.executable, str(Path(__file__).resolve()), '--source-run', str(source),
               '--run-dir', str(out), '--target-iteration', str(target), '--worker']
    with (out/'console.log').open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    plan.update(state='resuming', worker_pid=process.pid)
    write(out/'auto_continuation.json', plan)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    monitor_command = [sys.executable, '-u', '-m', 'deploy.deploy_mujoco.evaluate_him_checkpoint',
                       '--run-dir', str(out), '--watch']
    with (out/'checkpoint_monitor.log').open('x') as stream:
        monitor = subprocess.Popen(monitor_command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    write(out/'checkpoint_monitor_launch.json', {'pid':monitor.pid, 'CPU_only':True, 'hardware_access':False})
    code = process.wait()
    plan.update(state='completed' if code == 0 else 'failed', exit_code=code)
    write(out/'auto_continuation.json', plan)
    if code:
        raise RuntimeError(f'Continuation process exited with {code}; see console.log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--target-iteration', type=int, required=True)
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    source, out = args.source_run.resolve(), args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.background:
        if (out/'launch.json').exists():
            old = json.loads((out/'launch.json').read_text())
            if active(int(old['pid'])):
                raise ValueError('Continuation already running')
            if not (out/'orchestration_failure.json').exists():
                raise ValueError('Existing continuation is not a failed attempt')
            if (out/'auto_continuation.json').exists():
                previous = json.loads((out/'auto_continuation.json').read_text())
                if previous.get('worker_pid') and active(int(previous['worker_pid'])):
                    raise ValueError('Previous worker still active')
            if (out/'manifest.json').exists():
                raise ValueError('Failed worker already initialized; preserve its recorded run for review')
            archive = out/'attempts'/datetime.now().strftime('%Y%m%d_%H%M%S')
            archive.mkdir(parents=True)
            for path in list(out.iterdir()):
                if path.name != 'attempts':
                    shutil.move(str(path), str(archive/path.name))
        command = [sys.executable, str(Path(__file__).resolve()), '--source-run', str(source),
                   '--run-dir', str(out), '--target-iteration', str(args.target_iteration)]
        with (out/'orchestrator.log').open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        write(out/'launch.json', {'pid':process.pid, 'command':command})
        print(json.dumps({'scheduled_pid':process.pid, 'target_iteration':args.target_iteration,
                          'run_dir':str(out)}), flush=True)
        return
    try:
        if args.worker:
            worker(source, out, args.target_iteration)
        else:
            orchestrate(source, out, args.target_iteration)
    except BaseException as error:
        write(out/('failure.json' if args.worker else 'orchestration_failure.json'),
              {'error':repr(error), 'traceback':traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
