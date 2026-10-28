"""Offline HIM preflight: model, history, real-controller faults, and SDK serialization.

Never creates a DDS participant or publishes a command. Physical commissioning
records are deliberately not manufactured by this check.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from deploy.common.him_qualification import sha256, require_sim2sim
from deploy.common.go2_policy import build_observation, PolicyObservationHistory
from deploy.deploy_real.deploy_real_go2 import Go2RealConfig, Go2Controller, Go2State, Go2Mode, Go2DDSTransport, Go2Command
from deploy.deploy_real.go2_watchdog import CommandGate

ROOT=Path(LEGGED_GYM_ROOT_DIR)


def check(config):
    torch.set_num_threads(1)
    model=torch.jit.load(config.policy_path,map_location='cpu').eval()
    checks={}
    history=PolicyObservationHistory(config.data)
    q=config.default_joint_pos.copy()
    command=np.array([.3,-.2,.1],dtype=np.float32)
    frame=build_observation([.1,.2,.3],[0,0,-1],command,q,np.zeros(12),np.zeros(12),config.data)
    independent=np.concatenate((command*np.array([2.,2.,.25]),np.array([.1,.2,.3])*.25,[0,0,-1],np.zeros(36))).astype(np.float32)
    checks['training_order_and_scale']=bool(np.allclose(frame,independent,atol=1e-7))
    obs=history.append(frame)
    checks['first_history_zero']=bool(np.all(obs[45:]==0))
    timings=[]
    with torch.inference_mode():
        for index in range(220):
            start=time.perf_counter()
            result=model(torch.from_numpy(obs).unsqueeze(0))
            if index>=20:timings.append((time.perf_counter()-start)*1000)
            if result.shape!=(1,12) or not torch.isfinite(result).all():
                raise ValueError('HIM输出不是12个有限动作')
    checks['torchscript_270_to_12']=True
    checks['inference_p99_under_20ms']=bool(np.percentile(timings,99)<20.)
    # Deterministic fault stimuli isolate the controller guards from learned actions.
    def fresh():
        controller=Go2Controller(config,policy=lambda obs:torch.zeros((1,12)))
        controller.fsm.mode=Go2Mode.POLICY
        return controller
    def state(now=1.):
        return Go2State(now,np.array([1.,0,0,0]),np.zeros(3),q[config.sdk_to_policy].copy(),np.zeros(12))
    checks['nominal_controller']=fresh().step(1.,state()).mode is Go2Mode.POLICY
    checks['stale_state_stops']=fresh().step(1.2,state()).mode is Go2Mode.DAMPING
    controller=fresh();controller.step(1.,state());checks['late_loop_stops']=controller.step(1.08,state(1.08)).mode is Go2Mode.DAMPING
    controller=fresh();controller.policy=lambda obs:torch.full((1,12),float('nan'))
    checks['nan_action_stops']=controller.step(1.,state()).mode is Go2Mode.DAMPING
    controller=fresh();controller.policy=lambda obs:torch.ones((1,12))*10
    large_output=controller.step(1.,state())
    if config.software_motion_limits_enabled:
        checks['large_action_stops']=large_output.mode is Go2Mode.DAMPING
    else:
        checks['finite_action_passthrough']=bool(large_output.mode is Go2Mode.POLICY and
            np.allclose(large_output.target_pos_sdk,(q+2.5)[config.sdk_to_policy]))
    controller=fresh();controller.emergency_stop(1.,'offline injected stop')
    checks['emergency_latched']=controller.step(1.01,state(1.01)).mode is Go2Mode.DAMPING
    gate=CommandGate(config,2.)
    payload={'time':1.,'mode':'fix_stand','target':q[config.sdk_to_policy].tolist(),'kp':[5.]*12,'kd':[.5]*12}
    gate.accept(payload,1.,state())
    checks['watchdog_nominal']=gate.output(1.02,state(1.02)).mode is Go2Mode.FIX_STAND
    checks['watchdog_lease_stop']=gate.output(1.11,state(1.11)).mode is Go2Mode.DAMPING
    gate.accept(dict(payload,time=1.12),1.12,state(1.12))
    checks['watchdog_stop_latched']=gate.output(1.12,state(1.12)).mode is Go2Mode.DAMPING
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    transport=Go2DDSTransport.__new__(Go2DDSTransport)
    transport._low_cmd=unitree_go_msg_dds__LowCmd_()
    transport._init_low_cmd();transport._crc=CRC();transport._publish=True
    sent=[];transport._publisher=SimpleNamespace(Write=lambda msg:sent.append(msg.serialize()))
    target=q[config.sdk_to_policy]
    transport.send(Go2Command(Go2Mode.POLICY,target,config.stiffness[config.sdk_to_policy],config.damping[config.sdk_to_policy]))
    message=transport._low_cmd
    checks['official_sdk_crc']=bool(message.crc==CRC().Crc(message) and len(sent)==1)
    checks['sdk_joint_mapping']=bool(np.allclose([motor.q for motor in message.motor_cmd[:12]],target))
    checks['sdk_servo_mode']=all(m.mode==1 for m in message.motor_cmd[:12])
    checks['unused_motors_disabled']=all(m.kp==0 and m.kd==0 and m.tau==0 for m in message.motor_cmd[12:])
    checks['baseline_unchanged']=(sha256(ROOT/'deploy/pre_train/go2/motion.pt')=='a955b95e911e30f2db5d0435676c76dcf9b193a663adb5992a9b5abcb056773a' and
        sha256(ROOT/'deploy/deploy_real/configs/go2.yaml')=='03a291741ba5fe6d01ed9447f052dd1e96efea485268e50c12ad889af42cb184')
    qualification_error=None
    try:require_sim2sim(config);qualified=True
    except (OSError,ValueError,KeyError) as error:qualified=False;qualification_error=str(error)
    enforced=config.software_motion_limits_enabled
    return {'offline_only':True,'dds_initialized':False,'physical_robot_started':False,
            'software_motion_limits_enabled':enforced,'sim2sim_enforced':enforced,
            'interface_passed':all(checks.values()),'sim2sim_passed':qualified,'passed':all(checks.values()) and (qualified or not enforced),
            'checks':checks,'qualification_error':qualification_error,
            'latency_ms':{'median':float(np.median(timings)),'p99':float(np.percentile(timings,99)),'max':max(timings)},
            'policy_sha256':sha256(config.policy_path),'config_sha256':config.source_sha256,
            'field_readiness':'Physical stand/zero records required before keyboard movement' if enforced else
                'Physical walking remains unverified; historical records are retained, not used as entry gates'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'deploy/deploy_real/configs/go2_him.yaml')
    parser.add_argument('--output',type=Path,default=ROOT/'logs/go2_him/phase3_preflight/summary.json')
    args=parser.parse_args()
    report=check(Go2RealConfig.load(args.config,str(ROOT)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
