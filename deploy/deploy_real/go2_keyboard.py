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


def validate_readiness(path, mode, config, manifest, motion_seconds=None, walk_speed=.3, action_step_mode=None, left_speed=.3, speed_profile="legacy"):
    action_step_mode = commission.resolve_action_step_mode(action_step_mode, mode != 'zero' or (config is not None and config.data.get('format_version') == 2))
    if (walk_speed > .3 or left_speed > .3) and action_step_mode != "limit":
        raise ValueError("提高前进速度必须启用--action-step-mode limit")
    if walk_speed not in (.3,.5) or (walk_speed != .3 and mode != "all"):
        raise ValueError("W速度档仅用于all统一模式，允许0.30或0.50m/s")
    if motion_seconds not in (None,4,10) or (motion_seconds == 10 and mode != "all"):
        raise ValueError("10秒运动档仅用于all统一模式")
    if left_speed not in (.3,.4) or (left_speed != .3 and (mode != 'all' or motion_seconds is not None)):
        raise ValueError('A速度候选仅支持all模式0.40')
    kind = 'stand' if mode == 'zero' else 'zero'
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    pinned = manifest.get('readiness', {}).get(kind, {})
    if sha256(path) != pinned.get('sha256'):
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
                                  yaw_speed=.5 if speed_profile == 'uniform' else .3, turn_forward_speed=.5 if speed_profile == 'uniform' else .3 if mode in ('all', 'walk-turn') else 0.,
                                  keyboard_motion_seconds=motion_seconds if mode == 'all' else None, continuous=mode == 'all' and motion_seconds is None, walk_speed=walk_speed, action_step_mode=action_step_mode, left_speed=left_speed, speed_profile=speed_profile)


def commission_arguments(mode, network, readiness, motion_seconds=None, walk_speed=.3, action_step_mode=None, left_speed=.3, speed_profile="legacy", config_path=CONFIG):
    him = Go2RealConfig.load(config_path, str(ROOT)).data.get("format_version") == 2
    action_step_mode = commission.resolve_action_step_mode(action_step_mode, mode != "zero" or him)
    if (walk_speed > .3 or left_speed > .3) and action_step_mode != "limit":
        raise ValueError("提高前进速度必须启用--action-step-mode limit")
    if walk_speed not in (.3,.5) or (walk_speed != .3 and mode != "all"):
        raise ValueError("W速度档仅用于all统一模式，允许0.30或0.50m/s")
    if motion_seconds not in (None,4,10) or (motion_seconds == 10 and mode != "all"):
        raise ValueError("10秒运动档仅用于all统一模式")
    if left_speed not in (.3,.4) or (left_speed != .3 and (mode != 'all' or motion_seconds is not None)):
        raise ValueError('A速度候选仅支持all模式0.40')
    args = [network, '--config', str(config_path),
            '--policy-check' if mode == 'zero' else '--forward-check', str(readiness),
            '--start-standing', '--confirm-standing', '--release-mode',
            '--allow-body-movement', '--allow-policy-movement']
    if mode != 'zero':
        args += ['--keyboard', '--forward-speed', '0.4', '--keyboard-axis', {'all':'all', 'walk-turn':'yaw', 'backward':'backward', 'lateral':'lateral'}[mode],
                 '--yaw-speed', '0.3', '--turn-forward-speed', '0.3' if mode in ('all', 'walk-turn') else '0.0']
    if mode != 'zero':
        args += ['--action-step-mode', action_step_mode]
    if mode == 'all':
        args += ['--walk-speed',str(walk_speed)]
        args += ['--continuous'] if motion_seconds is None else ['--keyboard-motion-seconds',str(motion_seconds)]
    if left_speed != .3:
        args += ['--left-speed', str(left_speed)]
    if speed_profile == 'uniform':
        args += ['--speed-profile', 'uniform']
        args[args.index('--yaw-speed')+1] = '0.5'
        args[args.index('--turn-forward-speed')+1] = '0.5'
    return args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--mode', choices=('all', 'zero', 'walk-turn', 'backward', 'lateral'), default='all')
    parser.add_argument('--motion-seconds',type=int,choices=(4,10),default=None,help='可选短时运动预算；all默认持续运行至Z/Ctrl+C')
    parser.add_argument('--action-step-mode',choices=('reject','limit'),default=None,help='运动模式默认limit；reject保留显式对照')
    parser.add_argument('--walk-speed',type=float,choices=(.3,.5),default=None,help='uniform固定0.5；legacy中只改变W')
    parser.add_argument('--network', default='eno1')
    parser.add_argument('--readiness', type=Path, help='覆盖本机已验证的前置JSONL路径')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--check', action='store_true', help='离线检查；也是缺省行为')
    action.add_argument('--run', action='store_true', help='在前台确认后启动真机流程')
    parser.add_argument('--velocity-feedback', choices=('off', 'shadow', 'pi'), default='off',
                        help='off原控制；shadow记录速度；pi启用低速PI候选，不含足端力矩反馈')
    parser.add_argument('--velocity-frame', choices=('body', 'world'), help='现场验证后的速度坐标系')
    parser.add_argument('--velocity-readiness', type=Path, help='接管后连续速度观测JSONL；PI候选必需')
    parser.add_argument('--gait-restart', action='store_true', help='左移步态停滞后回站姿并重新起步一次')
    parser.add_argument('--left-speed', type=float, choices=(.3,.4), default=None, help='uniform固定0.4；legacy中只改变A')
    parser.add_argument('--speed-profile', choices=('legacy','uniform'), help='all默认uniform：前后0.5、侧移0.4、行进转向0.5/0.5；legacy保留历史对照')
    parser.add_argument('--velocity-source', choices=('sport','kinematic'), default='sport', help='kinematic使用关节与IMU估计速度，不依赖原厂速度话题')
    options = parser.parse_args(argv)
    try:
        config = Go2RealConfig.load(options.config, str(ROOT))
    except (OSError, ValueError, KeyError) as error:
        parser.error(f'配置加载失败：{error}')
    him = config.data.get('format_version') == 2
    if options.velocity_source == 'kinematic':
        if options.velocity_feedback == 'off' or options.velocity_frame == 'world':
            parser.error('运动学速度需要shadow或pi，坐标系固定body')
        options.velocity_frame = 'body'
    options.speed_profile = options.speed_profile or ('uniform' if not him and options.mode == 'all' and options.motion_seconds is None else 'legacy')
    uniform = options.speed_profile == 'uniform'
    options.walk_speed = options.walk_speed if options.walk_speed is not None else .5 if uniform else .3
    options.left_speed = options.left_speed if options.left_speed is not None else .4 if uniform else .3
    if uniform and (options.mode != 'all' or options.motion_seconds is not None or options.walk_speed != .5 or options.left_speed != .4 or options.gait_restart):
        parser.error('uniform固定前后0.5、侧移0.4、行进转向0.5/0.5，暂不叠加起步恢复；历史候选请显式--speed-profile legacy')
    if options.left_speed != .3 and (options.mode != 'all' or options.motion_seconds is not None or options.gait_restart or (options.velocity_feedback == 'pi' and not uniform)):
        parser.error('A速度对照仅支持all，不能叠加PI或起步恢复')
    if options.gait_restart and (options.mode != 'all' or options.motion_seconds is not None or options.velocity_feedback == 'pi'):
        parser.error('--gait-restart仅支持all持续控制，暂不与PI叠加')
    if options.velocity_feedback != 'off' and options.mode != 'all':
        parser.error('速度反馈仅支持all统一键盘模式')
    if options.velocity_feedback == 'pi' and (options.velocity_frame is None or options.velocity_readiness is None or (options.walk_speed != .3 and not uniform) or options.motion_seconds is not None):
        parser.error('PI候选需要--velocity-frame和--velocity-readiness，使用持续模式；新速度使用uniform配置')
    if options.velocity_readiness and options.velocity_feedback != 'pi':
        parser.error('--velocity-readiness仅用于PI候选')
    if options.velocity_frame and options.velocity_feedback == 'off':
        parser.error('--velocity-frame需要启用速度观测或PI')
    if options.walk_speed != .3 and options.mode != 'all':
        parser.error('--walk-speed 0.5仅用于all统一模式')
    if options.motion_seconds == 10 and options.mode != 'all':
        parser.error('--motion-seconds 10仅用于all统一模式')
    if options.network == 'lo' or not options.network.strip():
        parser.error('此入口仅用于实体网卡，不能使用lo')
    try:
        options.action_step_mode = commission.resolve_action_step_mode(options.action_step_mode, options.mode != 'zero' or him)
        manifest = json.loads(options.baseline.read_text(encoding='utf-8'))
        verify_baseline(config, manifest, options.config)
        kind = 'stand' if options.mode == 'zero' else 'zero'
        if not options.readiness and kind not in manifest.get('readiness', {}):
            raise ValueError("此模型尚无现场前置记录；请完成站姿/零速验收后通过--readiness提供，不能沿用旧模型记录")
        readiness = options.readiness or ROOT / manifest['readiness'][kind]['path']
        trial = validate_readiness(readiness, options.mode, config, manifest, options.motion_seconds, options.walk_speed, options.action_step_mode, options.left_speed, options.speed_profile)
        trial.controller.validate_policy()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(f'离线核对失败：{error}。历史日志不随仓库分发；可用--readiness提供匹配记录。')
    print(f'基线：{manifest["name"]}；PD {trial.policy_pd_limit:g}Nm，反馈停止 {trial.torque_limit:g}Nm。', flush=True)
    if options.mode == 'all':
        if options.motion_seconds is None:
            print('持续控制：接管后第8秒开放按键，无累计运动预算或自动结束时间；C受控降低退出，Z/Ctrl+C紧急终止。', flush=True)
        else:
            print(f'{trial.duration:g}秒流程：第8～{trial.keyboard_window_end:g}秒接受Q/W/E/A/S/D；累计运动最多{trial.keyboard_budget:g}秒，第{trial.lower_begin:g}秒开始降低。', flush=True)
        if options.velocity_feedback != 'pi':
            if uniform:
                print('W/S前后±0.50m/s；A/D侧移±0.40m/s；Q/E前进0.50m/s、转向±0.50rad/s。', flush=True)
            else:
                print(f'W前进{options.walk_speed:.2f}m/s；S后退0.30、A左移{options.left_speed:.2f}、D右移0.30m/s；Q/E前进0.30m/s加左右转0.30rad/s。', flush=True)
        if options.action_step_mode == 'limit':
            print(f'动作单步限幅{trial.controller.action_step_limit:g}；关节速度逐关节取URDF与配置交集：{trial.policy_velocity_limits.tolist()} rad/s（SDK顺序）。', flush=True)
        print('最后方向键生效，换方向先归零再加速；X/空格停走，C受控降低退出，Z/Ctrl+C紧急阻尼。', flush=True)
        print('看到按键窗口提示后再操作；松键最长0.6秒归零，请主动按X。', flush=True)
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
        print('HIM零速流程：5秒接管、2秒保持、2秒零速策略、4秒回站姿、4秒保持、8秒降低、5秒稳定。' if him else '16秒流程：5秒接管、2秒保持、2秒零速度策略、5秒降低、2秒稳定。', flush=True)
        print('零速度策略也可能迈步或漂移；本模式X/空格直接停止并阻尼。', flush=True)
    print('Z/Ctrl+C或第二终端停止进入阻尼并退出；阻尼取消位置支撑，机身会下沉。', flush=True)
    print('第二终端：python -m deploy.deploy_real.go2_watchdog --stop --domain-id 0', flush=True)
    arguments = commission_arguments(options.mode, options.network, readiness, options.motion_seconds, options.walk_speed, options.action_step_mode, options.left_speed, options.speed_profile, config_path=options.config)
    if him:
        print('HIM运动后C：制动2秒、回站姿4秒、保持4秒、降低8秒、稳定5秒，再阻尼收尾1秒。', flush=True)
    elif uniform:
        print('运动后C：制动2秒、过渡站姿2秒、保持4秒、降低8秒、稳定3秒，再阻尼收尾1秒。', flush=True)
    elif options.left_speed > .3:
        print('A=0.40对照档；用过该档后C先制动并平滑回站姿，再受控降低。X停步，Z急停。', flush=True)
    if options.gait_restart:
        arguments.append('--gait-restart')
        print('左移起步恢复已启用：关节步态持续静止时回站姿并重新起步一次；仍未起步则本次左移归零，X或其他方向重置尝试。', flush=True)
    if options.velocity_feedback != 'off':
        from deploy.deploy_real.go2_velocity_feedback import VelocityFeedback
        observer = VelocityFeedback(options.velocity_feedback, trial.controller.config.data['command_limits'], options.velocity_frame,
                                    source=options.velocity_source, config=config, speed_profile=options.speed_profile)
        if options.velocity_feedback == 'pi':
            from deploy.deploy_real.go2_velocity_feedback import validate_feedback_readiness
            try:
                validate_feedback_readiness(options.velocity_readiness, manifest['policy_sha256'], manifest['config_sha256'],
                    source=options.velocity_source, estimator_sha256=getattr(observer.source,'fingerprint',None),
                    speed_profile=options.speed_profile if uniform else None,
                    profile_sha256=observer.profile_sha256 if uniform else None)
            except (OSError, ValueError, KeyError, TypeError) as error:
                parser.error(f'速度反馈前置检查失败：{error}')
            arguments.extend(['--velocity-readiness', str(options.velocity_readiness.resolve())])
        arguments.extend(['--velocity-feedback', options.velocity_feedback, '--velocity-source', options.velocity_source])
        if options.velocity_frame:
            arguments.extend(['--velocity-frame', options.velocity_frame])
        print(f'反馈来源：{options.velocity_source}；运动学估计不等于独立实测速度。' if options.velocity_source == 'kinematic' else '反馈来源：原厂速度话题。', flush=True)
        print('速度观测：只记录，不修改策略命令。' if options.velocity_feedback == 'shadow' else
              f"PI候选：各轴目标上限{observer.profile['target_limits']}，修正上限{observer.profile['max_correction']}；内部指令不超过原上限，X清空积分，反馈失效触发原停止链路。", flush=True)
        if options.velocity_feedback == 'pi':
            print('C：制动2秒、站姿过渡2秒、保持4秒、降低8秒、稳定3秒，再阻尼收尾1秒；原控制与shadow的C流程不变。', flush=True)
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
    print('控制已退出；请现场确认机身停止情况。', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
