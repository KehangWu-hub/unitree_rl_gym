"""Read-only factory-walking reference; never releases mode or publishes LowCmd.

The onsite operator controls motion through the factory App/controller. This
sampler only records feedback and checks that the factory mode is normal.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from legged_gym import LEGGED_GYM_ROOT_DIR
from deploy.deploy_real.deploy_real_go2 import Go2RealConfig
from deploy.deploy_real.go2_commission import VerifiedTransport, mode_client, read_mode
from deploy.deploy_real.go2_velocity_feedback import SportVelocitySource

ROOT=Path(LEGGED_GYM_ROOT_DIR)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network',default='eno1')
    parser.add_argument('--duration',type=float,default=30.)
    parser.add_argument('--config',type=Path,default=ROOT/'deploy/deploy_real/configs/go2_him_fresh.yaml')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.network or args.network=='lo' or not 5<=args.duration<=60:
        parser.error('Use an actual network and a 5–60 second recording')
    config=Go2RealConfig.load(args.config,str(ROOT))
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(0,args.network)
    if read_mode(mode_client())!='normal':
        raise ValueError('原厂模式不是normal；只读程序不会恢复模式或起身，请由现场人员处理')
    transport=VerifiedTransport(config,publish=False)
    if transport._publisher is not None or transport._publish:
        raise RuntimeError('Read-only transport invariant failed')
    velocity=SportVelocitySource();velocity.start()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    last=None;count=0;peak=np.zeros(12);over=np.zeros(12,dtype=np.int64);moving_peak=np.zeros(12);moving_count=0
    joint_active_count=0;joint_active_peak=np.zeros(12)
    print('仅只读采样，不接管、不发电机命令。先静止约5秒，再由现场用原厂App低速前进，最后用原厂App停稳。',flush=True)
    begin=time.monotonic()
    try:
        with args.output.open('x',encoding='utf-8') as stream:
            while time.monotonic()-begin<args.duration:
                now=time.monotonic();state=transport.snapshot()
                if state is not None and state.timestamp!=last:
                    last=state.timestamp
                    f=state.feedback
                    row={'time':now,'elapsed':now-begin,'state_time':state.timestamp,
                         'quaternion_wxyz':state.quaternion_wxyz.tolist(),'angular_velocity':state.angular_velocity.tolist(),
                         'joint_pos_sdk':state.joint_pos_sdk.tolist(),'joint_vel_sdk':state.joint_vel_sdk.tolist(),'feedback':f}
                    try:row['factory_velocity']=velocity.snapshot(now)
                    except ValueError:row['factory_velocity']=None
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                    tau=np.abs(np.asarray(f['joint_tau_est_sdk']));peak=np.maximum(peak,tau);over+=tau>16.;count+=1
                    if np.max(np.abs(state.joint_vel_sdk))>1.:
                        joint_active_count+=1;joint_active_peak=np.maximum(joint_active_peak,tau)
                    speed=row['factory_velocity']
                    if speed is not None and np.linalg.norm(np.asarray(speed['velocity_raw'])[:2])>.05:
                        moving_peak=np.maximum(moving_peak,tau);moving_count+=1
                time.sleep(.004)
    finally:velocity.close()
    report={'read_only':True,'lowcmd_publisher_created':False,'motor_commands_sent':0,'factory_mode_at_start':'normal',
            'samples':count,'sport_velocity_above_005_samples':moving_count,'max_abs_feedback_torque_sdk':peak.tolist(),
            'samples_above_16nm_sdk':over.tolist(),'sport_velocity_labeled_peak_torque_sdk':moving_peak.tolist(),
            'joint_activity_samples':joint_active_count,'joint_activity_peak_torque_sdk':joint_active_peak.tolist(),
            'velocity_motion_disagreement':bool(joint_active_count and not moving_count),
            'scope':'Factory motion is performed by the onsite operator; this recording cannot validate HIM walking or recalibrate its stop threshold by itself.'}
    args.output.with_suffix('.summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if count else 1


if __name__=='__main__':raise SystemExit(main())
