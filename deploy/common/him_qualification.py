"""Bind HIM simulation evidence to the exact model and configuration."""
import hashlib
import json
from pathlib import Path

from legged_gym import LEGGED_GYM_ROOT_DIR
from deploy.common.go2_policy import resolve_path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


QUALIFICATION_SOURCES = (
    'deploy/common/go2_policy.py', 'deploy/common/him_qualification.py',
    'deploy/common/go2_velocity_control.py',
    'deploy/deploy_real/deploy_real_go2.py', 'deploy/deploy_real/go2_commission.py',
    'deploy/deploy_real/go2_keyboard.py', 'deploy/deploy_real/go2_velocity_feedback.py',
    'deploy/deploy_mujoco/validate_go2_him.py', 'deploy/deploy_mujoco/deploy_go2.py',
    'resources/robots/go2/mjcf/scene_flat.xml', 'resources/robots/go2/mjcf/go2.xml',
    'resources/robots/go2/urdf/go2.urdf',
    'legged_gym/scripts/evaluate_him.py', 'legged_gym/envs/go2_him/go2_him_env.py',
    'legged_gym/scripts/evaluate_him_candidate.py', 'legged_gym/envs/go2_him/acceptance.py',
    'legged_gym/envs/go2_him/retrain_config.py', 'legged_gym/envs/go2_him/retrain_env.py',
    'third_party/himloco_rsl_rl/rsl_rl/modules/bounded_him_actor_critic.py',
)


def current_sources(config=None):
    root = Path(LEGGED_GYM_ROOT_DIR)
    names = list(QUALIFICATION_SOURCES)
    package = Path(config.policy_path).parent if config is not None else root/'deploy/pre_train/go2_him'
    runtime = resolve_path(config.data.get('mujoco_runtime_config', str(root/'deploy/deploy_mujoco/configs/go2_him.yaml')), str(root)) if config is not None else str(root/'deploy/deploy_mujoco/configs/go2_him.yaml')
    names += [str((package/name).resolve().relative_to(root)) for name in ('deploy.yaml','provenance.json')]
    names.append(str(Path(runtime).resolve().relative_to(root)))
    return {name: sha256(root/name) for name in names}


def require_sim2sim(config):
    path = resolve_path(config.data['qualification_report'], LEGGED_GYM_ROOT_DIR)
    report = json.loads(Path(path).read_text())
    if report.get('simulation_only') is not True or report.get('passed') is not True:
        raise ValueError('HIM sim2sim及控制流程尚未通过')
    if report.get('policy_sha256') != sha256(config.policy_path) or report.get('config_sha256') != config.source_sha256:
        raise ValueError('HIM验证报告与当前模型/配置不匹配')
    files = report.get('source_sha256', {})
    required = current_sources(config)
    if not all(name in files for name in required):
        raise ValueError('HIM验证报告缺少控制代码/模型摘要')
    for name, digest in files.items():
        if sha256(Path(LEGGED_GYM_ROOT_DIR) / name) != digest:
            raise ValueError(f'HIM验证后文件发生改变: {name}')
    stages = ('raw', 'protected', 'zero', 'forward', 'comparison')
    if config.data.get('training_profile'):
        stages += ('isaac_acceptance', 'mujoco_acceptance')
    for stage in stages:
        evidence = report.get('evidence', {}).get(stage, {})
        evidence_path = Path(LEGGED_GYM_ROOT_DIR) / evidence.get('path', '')
        if not evidence_path.is_file() or sha256(evidence_path) != evidence.get('sha256'):
            raise ValueError(f'HIM验证证据缺失或变化: {stage}')
        data = json.loads(evidence_path.read_text())
        if data.get('passed') is not True:
            raise ValueError(f'HIM验证未通过: {stage}')
        if stage in ('isaac_acceptance','mujoco_acceptance'):
            if data.get('policy_sha256') != report['policy_sha256'] or data.get('command_profile') != 'keyboard':
                raise ValueError(f'HIM性能证据范围或模型不匹配: {stage}')
            continue
        if stage != 'comparison' and (data.get('policy_sha256') != report['policy_sha256'] or
                data.get('config_sha256') != config.source_sha256 or data.get('source_sha256') != files):
            raise ValueError(f'HIM验证证据版本不一致: {stage}')
    return report
