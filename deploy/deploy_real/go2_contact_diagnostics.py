"""只读采集CRC有效LowState的足端原始字段和IMU；不创建电机发布器。"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from deploy.deploy_real.deploy_real_go2 import Go2RealConfig, quaternion_to_roll_pitch
from deploy.deploy_real.go2_commission import VerifiedTransport
from deploy.deploy_real.go2_keyboard import CONFIG, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', required=True)
    parser.add_argument('--duration', type=float, default=3.)
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/go2_real/commission/contact_readonly.json')
    args = parser.parse_args()
    if not np.isfinite(args.duration) or not 1 <= args.duration <= 10:
        parser.error('只读时长必须为1～10秒')
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    config = Go2RealConfig.load(CONFIG, str(ROOT))
    ChannelFactoryInitialize(config.data['dds']['domain_id'], args.network)
    transport = VerifiedTransport(config, publish=False)
    rows = []
    last_timestamp = None
    start = time.monotonic()
    while time.monotonic() - start < args.duration:
        state = transport.snapshot()
        if state is not None and state.timestamp != last_timestamp:
            last_timestamp = state.timestamp
            rows.append({'timestamp': state.timestamp, 'quaternion_wxyz': state.quaternion_wxyz.tolist(),
                         'joint_pos_sdk': state.joint_pos_sdk.tolist(), 'joint_vel_sdk': state.joint_vel_sdk.tolist(),
                         'angular_velocity': state.angular_velocity.tolist(), 'feedback': state.feedback})
        time.sleep(.005)
    summary = {'read_only': True, 'network': args.network, 'frames': len(rows),
               'scope': '当前姿态下原始SDK字段；单位、足序和校准未验证，全零不能解释为无承重。'}
    if rows:
        for key in ('foot_force_raw', 'foot_force_est_raw', 'imu_accelerometer_raw'):
            values = np.asarray([r['feedback'][key] for r in rows])
            summary[key] = {'min': values.min(axis=0).tolist(), 'max': values.max(axis=0).tolist(),
                            'mean': values.mean(axis=0).tolist(), 'all_zero': bool(np.all(values == 0))}
        summary['roll_pitch_rad'] = list(quaternion_to_roll_pitch(rows[-1]['quaternion_wxyz']))
        summary['max_joint_velocity_rad_s'] = float(np.max(np.abs([r['joint_vel_sdk'] for r in rows])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'summary': summary, 'rows': rows}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if rows else 1


if __name__ == '__main__':
    raise SystemExit(main())
