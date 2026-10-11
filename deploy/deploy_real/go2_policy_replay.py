"""离线回放commission日志；不初始化DDS，不发布电机命令。"""
import argparse
import json
from pathlib import Path

import numpy as np

from deploy.deploy_real.deploy_real_go2 import Go2Controller, Go2RealConfig, Go2State


def recorded_state(row):
    return Go2State(row['state_time'], np.asarray(row['quaternion_wxyz']),
                    np.asarray(row['angular_velocity']), np.asarray(row['joint_pos_sdk']),
                    np.asarray(row['joint_vel_sdk']))


def replay(rows, config):
    controller = Go2Controller(config)
    count, errors = 0, []
    for row in rows:
        if row['mode'] != 'policy':
            continue
        n = row['feedback']['policy_inference_count']
        if n == count:
            continue
        if n != count + 1:
            raise ValueError('推理序号不连续，不能准确回放历史动作')
        state = recorded_state(row)
        controller.set_velocity_command(row['velocity_command'])
        q, dq = controller._policy_joint_state(state)
        action, target = controller._policy_action(state, q, dq)
        # 初始接管混合阶段不参与误差比较；仍回放以重建last_action。
        if row['feedback']['test_elapsed'] >= 7 + row['feedback']['policy_entry_blend_seconds']:
            sdk = controller._to_sdk(target)
            qs, dqs = state.joint_pos_sdk, state.joint_vel_sdk
            cap = row['feedback']['policy_pd_limit_nm']
            pd = np.clip(20 * (sdk - qs) - .5 * dqs, -cap, cap)
            limited = qs + (pd + .5 * dqs) / 20
            errors.append(float(np.max(np.abs(limited - np.asarray(row['target_pos_sdk'])))))
        controller.last_action = action
        count = n
    if not count or not errors:
        raise ValueError('缺少可核对的策略推理记录')
    error = max(errors)
    result = {'offline_only': True, 'inferences_replayed': count,
              'max_valid_target_replay_error_rad': error,
              'historical_outputs_match': error <= 1e-5,
              'recorded_stop_reason': next((r['reason'] for r in rows if r.get('reason')), '')}
    stop = next((r for r in rows if r.get('reason')), None)
    if stop and stop['reason'] == '目标关节角超过安全限制' and error <= 1e-5:
        # 旧日志缺少原始故障动作；这是按末次指令和停止帧反馈推算，不能冒充原始记录。
        state = recorded_state(stop)
        q, dq = controller._policy_joint_state(state)
        action, target = controller._policy_action(state, q, dq)
        indices = np.flatnonzero((target < controller.safety.joint_lower) |
                                (target > controller.safety.joint_upper))
        result['fault_source'] = 'reconstructed_from_stop_state_and_last_command'
        result['reconstructed_violating_joints'] = [
            {'joint': config.data['policy']['joint_names'][int(i)],
             'target_rad': float(target[i]), 'actual_rad': float(q[i]),
             'lower_rad': float(controller.safety.joint_lower[i]),
             'upper_rad': float(controller.safety.joint_upper[i])} for i in indices]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = Go2RealConfig.load(root / 'deploy/deploy_real/configs/go2.yaml', str(root))
    rows = [json.loads(line) for line in args.log.read_text(encoding='utf-8').splitlines() if line.strip()]
    result = replay(rows, config)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + '\n',encoding='utf-8')
    print(rendered)


if __name__ == '__main__':
    main()
