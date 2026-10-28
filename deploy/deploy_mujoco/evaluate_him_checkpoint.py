"""CPU-only checkpoint monitoring on flat MuJoCo ground; never deploys a policy."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import time

import torch
import yaml

from third_party.himloco_rsl_rl.rsl_rl.modules import HIMActorCritic, BoundedHIMActorCritic
from deploy.common.him_qualification import sha256
from deploy.deploy_real.deploy_real_go2 import Go2RealConfig
from deploy.deploy_mujoco.gait_metrics import COMMANDS, measure
from deploy.deploy_mujoco.validate_go2_him import ROOT


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def evaluate(run, iteration, keys):
    out = run/'checkpoint_evaluations'/str(iteration)
    out.mkdir(parents=True, exist_ok=True)
    train = yaml.safe_load((run/'train.yaml').read_text())
    env = yaml.safe_load((run/'env.yaml').read_text())
    checkpoint = run/f'model_{iteration}.pt'
    saved = torch.load(checkpoint, map_location='cpu')
    assert saved['iter'] == iteration
    cls = {'HIMActorCritic': HIMActorCritic, 'BoundedHIMActorCritic': BoundedHIMActorCritic}[train['runner']['policy_class_name']]
    with contextlib.redirect_stdout(io.StringIO()):
        model = cls(env['env']['num_observations'], env['env']['num_privileged_obs'],
                    env['env']['num_one_step_observations'], env['env']['num_actions'], **train['policy']).eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    spec = importlib.util.spec_from_file_location('checkpoint_him_export', ROOT/'legged_gym/envs/go2_him/export.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    policy = Path(exporter.export_him_policy(model, out/'exported'))
    torch.manual_seed(91)
    history = torch.randn(64, 270)*.2
    with torch.no_grad():
        error = float((model.act_inference(history)-torch.jit.load(str(policy))(history)).abs().max())
    if error != 0:
        raise ValueError(f'Checkpoint export mismatch: {error}')
    data = yaml.safe_load((ROOT/'deploy/deploy_real/configs/go2_him_reward_restore_v2.yaml').read_text())
    exported = yaml.safe_load((run/'exported/params/deploy.yaml').read_text())
    # Keep the full simulator controller config, overlay the actual training contract.
    for name in ('format_version', 'robot', 'step_dt', 'policy', 'observations', 'actions', 'control',
                 'commands', 'training_profile', 'training_source'):
        if name in exported:
            data[name] = exported[name]
    data['policy_path'] = str(policy)
    data['software_motion_limits'] = False
    data['hardware_deployment_blocked'] = True
    data['hardware_deployment_block_reason'] = '断点评估仅用于离线MuJoCo，不允许真机接管'
    config_path = out/'config.yaml'
    config_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    cfg = Go2RealConfig.load(config_path, str(ROOT))
    result = {'simulation_only': True, 'dds_initialized': False, 'iteration': iteration,
              'checkpoint_sha256': sha256(checkpoint), 'policy_sha256': sha256(policy),
              'export_max_error': error, 'hardware_deployment_blocked': True,
              'scope': 'Nominal flat MuJoCo only; does not validate stair climbing or hardware.',
              'overall_gait_passed': None, 'cases': {}}
    for key in keys:
        result['cases'][key] = measure(cfg, key, out/key)
        write_json(out/'results.json', result)
    result['complete'] = True
    result['body_checks_passed'] = all(r['flow']['passed'] for r in result['cases'].values())
    # Report foot metrics; never turn body-only checks into a gait qualification.
    write_json(out/'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--iteration', type=int)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--keys', nargs='+', choices=tuple(COMMANDS), default=list(COMMANDS))
    args = parser.parse_args()
    if args.watch == (args.iteration is not None):
        parser.error('Choose exactly one of --watch or --iteration')
    torch.set_num_threads(1)
    run = args.run_dir.resolve()
    if not args.watch:
        report = evaluate(run, args.iteration, args.keys)
        print(json.dumps({'iteration': args.iteration, 'complete': True,
                          'body_checks_passed': report['body_checks_passed']}, ensure_ascii=False), flush=True)
        return
    out = run/'checkpoint_evaluations'
    out.mkdir(exist_ok=True)
    started = time.monotonic()
    completed, errors = {}, {}
    while time.monotonic()-started < 24*3600:
        candidates = sorted((p for p in run.glob('model_*.pt') if time.time()-p.stat().st_mtime > 2),
                            key=lambda p: int(p.stem.split('_')[-1]))
        for path in candidates:
            iteration = int(path.stem.split('_')[-1])
            manifest_path = run/'manifest.json'
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            if iteration <= manifest.get('start_iteration', 0) or str(iteration) in completed or str(iteration) in errors:
                continue
            try:
                report = evaluate(run, iteration, args.keys)
                completed[str(iteration)] = {'body_checks_passed': report['body_checks_passed'],
                                             'results': str((out/str(iteration)/'results.json').relative_to(run))}
                print(json.dumps({'iteration': iteration, **completed[str(iteration)]}), flush=True)
            except Exception as error:
                errors[str(iteration)] = repr(error)
                print(json.dumps({'iteration': iteration, 'error': repr(error)}), flush=True)
            write_json(out/'index.json', {'simulation_only': True, 'completed': completed, 'errors': errors,
                                          'automatic_hardware_deployment': False, 'overall_gait_passed': None})
        progress_path = run/'progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        if progress.get('state') == 'training_finished_pending_evaluation':
            final = str(progress['iteration'])
            if final in completed or final in errors:
                return
        if (run/'failure.json').exists():
            return
        time.sleep(5)
    raise RuntimeError('Checkpoint monitor reached 24-hour limit')


if __name__ == '__main__':
    main()
