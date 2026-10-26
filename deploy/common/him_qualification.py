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
    'deploy/deploy_mujoco/configs/go2_him.yaml',
    'deploy/pre_train/go2_him/deploy.yaml', 'deploy/pre_train/go2_him/provenance.json',
    'resources/robots/go2/mjcf/scene_flat.xml', 'resources/robots/go2/mjcf/go2.xml',
    'resources/robots/go2/urdf/go2.urdf',
    'legged_gym/scripts/evaluate_him.py', 'legged_gym/envs/go2_him/go2_him_env.py',
)


def current_sources():
    return {name: sha256(Path(LEGGED_GYM_ROOT_DIR) / name) for name in QUALIFICATION_SOURCES}


def require_sim2sim(config):
    path = resolve_path(config.data['qualification_report'], LEGGED_GYM_ROOT_DIR)
    report = json.loads(Path(path).read_text())
    if report.get('simulation_only') is not True or report.get('passed') is not True:
        raise ValueError('HIM sim2sim及控制流程尚未通过')
    if report.get('policy_sha256') != sha256(config.policy_path) or report.get('config_sha256') != config.source_sha256:
        raise ValueError('HIM验证报告与当前模型/配置不匹配')
    files = report.get('source_sha256', {})
    required = QUALIFICATION_SOURCES
    if not all(name in files for name in required):
        raise ValueError('HIM验证报告缺少控制代码/模型摘要')
    for name, digest in files.items():
        if sha256(Path(LEGGED_GYM_ROOT_DIR) / name) != digest:
            raise ValueError(f'HIM验证后文件发生改变: {name}')
    for stage in ('raw', 'protected', 'zero', 'forward', 'comparison'):
        evidence = report.get('evidence', {}).get(stage, {})
        evidence_path = Path(LEGGED_GYM_ROOT_DIR) / evidence.get('path', '')
        if not evidence_path.is_file() or sha256(evidence_path) != evidence.get('sha256'):
            raise ValueError(f'HIM验证证据缺失或变化: {stage}')
        data = json.loads(evidence_path.read_text())
        if data.get('passed') is not True:
            raise ValueError(f'HIM验证未通过: {stage}')
        if stage != 'comparison' and (data.get('policy_sha256') != report['policy_sha256'] or
                data.get('config_sha256') != config.source_sha256 or data.get('source_sha256') != files):
            raise ValueError(f'HIM验证证据版本不一致: {stage}')
    return report
