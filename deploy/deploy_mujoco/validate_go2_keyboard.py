"""用真机同一控制流程检查键盘运动及方向切换；只运行MuJoCo，不初始化DDS。"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from deploy.deploy_mujoco.deploy_go2 import joint_addresses
from deploy.deploy_real.deploy_real_go2 import Go2Mode, Go2State
from deploy.deploy_real.go2_keyboard import BASELINE, CONFIG, ROOT, sha256, validate_readiness, verify_baseline
from deploy.deploy_real.deploy_real_go2 import Go2RealConfig
from deploy.deploy_real.go2_commission import assess_keyboard


def simulate(mode, key, config, manifest, readiness, initial, stop_time, model_dynamics=False,
             min_tracking_ratio=.5, key_schedule=None):
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'resources/robots/go2/mjcf/scene_flat.xml'))
    model.opt.timestep = .002
    if not model_dynamics:
        model.dof_damping[6:] = 0
        model.dof_armature[6:] = 0
        model.dof_frictionloss[6:] = 0
        model.geom_friction[:, 0] = 1.
    pos, vel, actuator = joint_addresses(model, config.data['policy']['sdk_joint_names'])
    data = mujoco.MjData(model)
    q = np.tile([0., .8, -1.5], 4) if initial is None else np.asarray(initial['joint_pos_sdk'])
    if initial is not None:
        data.qpos[3:7] = initial['quaternion_wxyz']
    data.qpos[pos] = q
    data.qpos[2] = .33
    mujoco.mj_forward(model, data)
    for _ in range(1500):
        data.ctrl[actuator] = np.clip(40 * (q - data.qpos[pos]) - data.qvel[vel], -8, 8)
        mujoco.mj_step(model, data)
    trial = validate_readiness(readiness, mode, config, manifest)
    last_tau = data.ctrl[actuator].copy()

    def state(now, tick):
        result = Go2State(now, data.qpos[3:7].copy(), data.qvel[3:6].copy(),
                          data.qpos[pos].copy(), data.qvel[vel].copy())
        result.feedback = {'joint_tau_est_sdk': last_tau.tolist(), 'state_tick': tick}
        return result

    trial.arm(1., state(1., 0))
    start = None
    end = None
    residual_speeds = []
    zero_commands = []
    axis = 0 if mode == 'backward' else 1
    direction = -1 if key in ('s', 'd') else 1
    commanded_distance = 0.
    motion_distance = 0.
    moving_speeds = []
    moving_times = []
    reason = ''
    rows = []
    commands = []
    active_keys = set()
    schedule = key_schedule or [(9., key)]
    for tick in range(int(trial.duration / .002) + 1):
        elapsed = tick * .002
        now = 1. + elapsed
        selected = next((item[1] for item in reversed(schedule) if elapsed >= item[0]), None)
        if selected is not None and 9. <= elapsed < stop_time and tick % 20 == 0:
            trial.feed_key(selected, now)
            if trial.keyboard_event and trial.keyboard_event['accepted']:
                active_keys.add(selected)
        if stop_time <= elapsed < stop_time + .002:
            trial.feed_key('x', now)
        command = trial.output(now, state(now, tick))
        if trial.phase(now)[0] == 'policy':
            if start is None:
                start = data.qpos[:2].copy()
            end = data.qpos[:2].copy()
            v = trial.controller.command
            commands.append(v.copy())
            requested_speed = float(np.linalg.norm(v[:2])) if mode == 'all' else direction * float(v[axis])
            if requested_speed > 0:
                rotation = data.xmat[model.body('base_link').id].reshape(3, 3)
                body_velocity = rotation.T @ data.qvel[:3]
                signed_speed = float(np.dot(body_velocity[:2], v[:2])/requested_speed) if mode == 'all' else direction * float(body_velocity[axis])
                commanded_distance += requested_speed * model.opt.timestep
                motion_distance += signed_speed * model.opt.timestep
                moving_speeds.append(signed_speed)
                moving_times.append(elapsed)
            if elapsed >= stop_time + 2.8:
                residual_speeds.append(float(np.linalg.norm(data.qvel[:2])))
                zero_commands.append(float(np.max(np.abs(trial.controller.command))))
        if mode == 'all' and (tick % 10 == 0 or command.mode is Go2Mode.DAMPING):
            rows.append({'time': now, 'mode':command.mode.value, 'reason':command.reason,
                         'joint_pos_sdk':data.qpos[pos].tolist(), 'joint_vel_sdk':data.qvel[vel].tolist(),
                         'quaternion_wxyz':data.qpos[3:7].tolist(), 'velocity_command':trial.controller.command.tolist(),
                         'feedback':{'stand_profile':'keyboard29', 'test_elapsed':elapsed,
                                     'keyboard_axis':'all', 'keyboard_event':trial.keyboard_event,
                                     'keyboard_forward_seconds':trial.keyboard_used,
                                     'forward_speed_mps':.3, 'lateral_speed_mps':.3,
                                     'yaw_speed_rad_s':.3, 'turn_forward_speed_mps':.3}})
        last_tau = command.kp_sdk * (command.target_pos_sdk - data.qpos[pos]) - command.kd_sdk * data.qvel[vel]
        data.ctrl[actuator] = last_tau
        if command.mode is Go2Mode.DAMPING:
            reason = command.reason
            break
        mujoco.mj_step(model, data)
    delta = end - start if start is not None else np.zeros(2)
    signed_distance = motion_distance if mode == 'all' else -delta[0] if mode == 'backward' else delta[1] * (1 if key == 'a' else -1)
    error = float(np.max(np.abs(data.qpos[pos] - trial.lowered)))
    residual = max(residual_speeds, default=float('inf'))
    flow_passed = (reason == '测试时长结束' and elapsed >= 29.
                   and residual < .02 and bool(zero_commands) and max(zero_commands) == 0.)
    tracking_ratio = motion_distance / commanded_distance if commanded_distance > 0 else 0.
    # 旧1cm判据只能确定方向，不能证明侧移效果。此门槛是明确的电脑端效果判据。
    motion_passed = commanded_distance > 0 and tracking_ratio >= min_tracking_ratio
    log_assessment = assess_keyboard(rows, config) if mode == 'all' else None
    command_checks = (bool(commands) and np.max(np.abs(commands)) <= .30001
                      and trial.keyboard_used <= 4.06 and bool(zero_commands) and max(zero_commands) == 0.
                      and log_assessment.get('command_checks_passed',False)) if mode == 'all' else True
    if mode == 'all':
        flow_passed = flow_passed and command_checks
    passed = flow_passed and motion_passed
    last_speeds = [speed for timestamp, speed in zip(moving_times, moving_speeds)
                   if timestamp >= moving_times[-1] - 1.] if moving_times else []
    return {'simulation_only': True, 'mode': mode, 'key': key, 'passed': bool(passed),
            'flow_passed': bool(flow_passed), 'motion_passed': bool(motion_passed),
            'command_checks_passed':bool(command_checks), 'log_assessment':log_assessment,
            'key_schedule':schedule, 'accepted_direction_keys':sorted(active_keys),
            'direction_only_passed': bool(signed_distance > .01),
            'commanded_axis_distance_m': commanded_distance,
            'moving_body_axis_distance_m': motion_distance, 'distance_tracking_ratio': tracking_ratio,
            'min_tracking_ratio': min_tracking_ratio,
            'last_moving_second_mean_axis_speed_m_s': float(np.mean(last_speeds)) if last_speeds else None,
            'dynamics': 'original_mjcf' if model_dynamics else 'training_overrides',
            'reason': reason, 'elapsed_seconds': elapsed, 'policy_displacement_xy_m': delta.tolist(),
            'residual_planar_speed_m_s': residual, 'final_tracking_error_rad': error,
            'settle_tracking_tolerance_rad': trial.settle_tracking_tolerance,
            'passes_previous_0_15_rad_criterion': error <= .15,
            'motion_budget_used_seconds': trial.keyboard_used,
            'final_height_m': float(data.qpos[2]),
            'scope': '完整接管、策略、X归零及降低；流程按控制器自身保护评价，日志分析另核对真机卧倒姿态。不模拟退出后的一秒阻尼，不代表真机通过'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unified', action='store_true', help='检查统一Q/W/E/A/S/D及跨方向切换')
    parser.add_argument('--initial-log', type=Path, help='可选：读取真实站姿记录第一行')
    parser.add_argument('--readiness', type=Path)
    parser.add_argument('--model-dynamics', action='store_true', help='保留MJCF阻尼、惯量和摩擦；这些值不代表实测真机参数')
    parser.add_argument('--stop-time', type=float, choices=(10.5, 13.3), default=10.5,
                        help='9秒开始按键；10.5秒测试X，13.3秒覆盖4秒预算耗尽')
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/go2_real/commission/keyboard_new_axes_mujoco.json')
    parser.add_argument('--min-tracking-ratio', type=float, default=.5,
                        help='电脑端运动效果门槛：运动窗口内实际/指令距离，默认0.5；不是原厂标准')
    args = parser.parse_args()
    if not np.isfinite(args.min_tracking_ratio) or not 0 < args.min_tracking_ratio <= 1:
        parser.error('--min-tracking-ratio必须在(0, 1]范围内')
    manifest = json.loads(BASELINE.read_text())
    config = Go2RealConfig.load(CONFIG, str(ROOT))
    verify_baseline(config, manifest)
    readiness = args.readiness or ROOT / manifest['readiness']['zero']['path']
    initial = json.loads(args.initial_log.read_text().splitlines()[0]) if args.initial_log else None
    if args.unified:
        cases = [(key, [(9.,key)], args.stop_time) for key in ('w','s','a','d','q','e')]
        cases += [('switch_all', [(9.,'w'),(9.7,'a'),(10.4,'q'),(11.1,'s'),(11.8,'d'),(12.5,'e')],12.9),
                  ('reverse', [(9.,'w'),(9.8,'s'),(10.6,'w')],11.4),
                  ('turn_reverse', [(9.,'q'),(9.8,'e')],10.5)]
        reports = [simulate('all', name, config, manifest, readiness, initial, stop, args.model_dynamics,
                            args.min_tracking_ratio, schedule) for name, schedule, stop in cases]
    else:
        reports = [simulate(mode, key, config, manifest, readiness, initial, args.stop_time, args.model_dynamics,
                            args.min_tracking_ratio)
                   for mode, key in [('backward', 's'), ('lateral', 'a'), ('lateral', 'd')]]
    result = {'simulation_only':True, 'flow_passed':all(row['flow_passed'] for row in reports),
              'command_checks_passed':all(row['command_checks_passed'] for row in reports),
              'motion_passed':all(row['motion_passed'] for row in reports), 'policy_sha256': sha256(config.policy_path), 'config_sha256': sha256(CONFIG),
              'initial_log': str(args.initial_log) if args.initial_log else None,
              'stop_time': args.stop_time, 'cases': reports, 'passed': all(row['passed'] for row in reports)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
