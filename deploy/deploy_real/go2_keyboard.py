"""Go2键盘控制的简短入口；默认离线检查，--run才启动真机流程。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from legged_gym import LEGGED_GYM_ROOT_DIR
from deploy.deploy_real.deploy_real_go2 import Go2RealConfig
from deploy.deploy_real import go2_commission as commission


ROOT = Path(LEGGED_GYM_ROOT_DIR)
CONFIG = ROOT / 'deploy/deploy_real/configs/go2.yaml'
BASELINE = ROOT / 'deploy/deploy_real/configs/go2_verified_baseline.json'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_baseline(config, manifest, config_path=CONFIG):
    if sha256(config.policy_path) != manifest['policy_sha256']:
        raise ValueError('策略与已验证基线不一致；请先核对版本和验证记录')
    if sha256(config_path) != manifest['config_sha256']:
        raise ValueError('配置与已验证基线不一致；请先核对改动和验证记录')


def validate_readiness(path, mode, config, manifest):
    kind = 'stand' if mode == 'zero' else 'zero'
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    pinned = manifest['readiness'][kind]
    if sha256(path) != pinned['sha256']:
        stand_rows = [row for row in rows if row.get('mode') == 'fix_stand']
        policy_rows = [row for row in rows if row.get('mode') == 'policy']
        matching_stand = kind == 'stand' and bool(stand_rows) and all(
            row.get('feedback', {}).get('control_config_sha256') == manifest['config_sha256']
            for row in stand_rows)
        matching_zero = kind == 'zero' and bool(policy_rows) and all(
                row.get('feedback', {}).get('policy_sha256') == manifest['policy_sha256']
                and row.get('feedback', {}).get('policy_config_sha256') == manifest['config_sha256']
                for row in policy_rows)
        if not (matching_stand or matching_zero):
            raise ValueError('前置记录缺少匹配基线的模型/配置标记；不能沿用未知版本记录')
    # 与实际执行共用判据，包括持续时间、曲线和PD/反馈参数匹配。
    return commission.PolicyCheck(config, str(path), forward=mode != 'zero',
                                  forward_speed=.4 if mode != 'zero' else .1,
                                  keyboard=mode != 'zero', keyboard_axis={'all':'all', 'walk-turn':'yaw', 'backward':'backward', 'lateral':'lateral'}.get(mode, 'forward'),
                                  yaw_speed=.3, turn_forward_speed=.3 if mode in ('all', 'walk-turn') else 0.)


def commission_arguments(mode, network, readiness):
    args = [network, '--config', str(CONFIG),
            '--policy-check' if mode == 'zero' else '--forward-check', str(readiness),
            '--start-standing', '--confirm-standing', '--release-mode',
            '--allow-body-movement', '--allow-policy-movement']
    if mode != 'zero':
        args += ['--keyboard', '--forward-speed', '0.4', '--keyboard-axis', {'all':'all', 'walk-turn':'yaw', 'backward':'backward', 'lateral':'lateral'}[mode],
                 '--yaw-speed', '0.3', '--turn-forward-speed', '0.3' if mode in ('all', 'walk-turn') else '0.0']
    return args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('all', 'zero', 'walk-turn', 'backward', 'lateral'), default='all')
    parser.add_argument('--network', default='eno1')
    parser.add_argument('--readiness', type=Path, help='覆盖本机已验证的前置JSONL路径')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--check', action='store_true', help='离线检查；也是缺省行为')
    action.add_argument('--run', action='store_true', help='在前台确认后启动真机流程')
    options = parser.parse_args(argv)
    if options.network == 'lo' or not options.network.strip():
        parser.error('此入口仅用于实体网卡，不能使用lo')
    try:
        manifest = json.loads(BASELINE.read_text(encoding='utf-8'))
        config = Go2RealConfig.load(CONFIG, str(ROOT))
        verify_baseline(config, manifest)
        kind = 'stand' if options.mode == 'zero' else 'zero'
        readiness = options.readiness or ROOT / manifest['readiness'][kind]['path']
        trial = validate_readiness(readiness, options.mode, config, manifest)
        trial.controller.validate_policy()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(f'离线核对失败：{error}。历史日志不随仓库分发；可用--readiness提供匹配记录。')
    print(f'基线：{manifest["name"]}；PD {trial.policy_pd_limit:g}Nm，反馈停止 {trial.torque_limit:g}Nm。', flush=True)
    if options.mode == 'all':
        print('29秒流程：第8～20秒接受Q/W/E/A/S/D；累计运动最多4秒，第22秒开始降低。', flush=True)
        print('W/S前后、A/D左右：0.30m/s；Q/E前进0.30m/s加左右转0.30rad/s。', flush=True)
        print('最后方向键生效，换方向先归零再加速；X/空格归零，Z/Ctrl+C阻尼。', flush=True)
        print('看到按键窗口提示后再操作；松键最长0.6秒归零，请主动按X。', flush=True)
        print('统一入口W与X已有短时现场确认；跨方向切换仍待验证。', flush=True)
    elif options.mode == 'walk-turn':
        print('29秒流程：5秒接管、2秒保持、1秒零速；第8～20秒接受Q/E；第22秒开始降低。', flush=True)
        print('Q/E：前进0.30m/s加左右转0.30rad/s；累计运动最多4秒。X/空格归零。', flush=True)
        print('只在看到“按住Q/E控制yaw”后按键；松键最长0.6秒归零，请主动按X。', flush=True)
    elif options.mode in ('backward', 'lateral'):
        keys = 'S' if options.mode == 'backward' else 'A/D'
        motion = '后退0.30m/s' if options.mode == 'backward' else '左右侧移0.30m/s'
        print('受限短时流程：29秒；第8～20秒接受键盘，累计运动最多4秒，第22秒开始降低。', flush=True)
        print(f'{keys}：{motion}；看到按键提示后再按。X/空格归零，松键最长0.6秒归零。', flush=True)
        if options.mode == 'lateral':
            print('左右侧移各有短时现场确认；此前停滞的触发原因和重复可靠性仍未确定。', flush=True)
    else:
        print('16秒流程：5秒接管、2秒保持、2秒零速度策略、5秒降低、2秒稳定。', flush=True)
        print('零速度策略也可能迈步或漂移；本模式X/空格直接停止并阻尼。', flush=True)
    print('随后1秒阻尼；Z/Ctrl+C或第二终端停止进入阻尼。结束后不自动恢复normal。', flush=True)
    print('第二终端：python -m deploy.deploy_real.go2_watchdog --stop --domain-id 0', flush=True)
    arguments = commission_arguments(options.mode, options.network, readiness)
    if not options.run:
        print('离线检查通过；没有初始化DDS或发送电机命令。加--run才启动。', flush=True)
        return 0
    if not sys.stdin.isatty():
        parser.error('--run必须使用前台交互终端')
    print('确认原厂已站稳、场地平坦无人、有人观察且停止终端可用。', flush=True)
    if input('输入START开始接管；其他输入取消：').strip() != 'START':
        print('已取消，未初始化DDS。')
        return 0
    print('开始接管；暂时不要操作App。', flush=True)
    commission.run(commission.parse_args(arguments), config)
    print('流程已退出；请现场确认卧倒和停止情况。日志正常结束不等于物理效果通过。', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
