"""Replay recorded HIM inputs and commands offline; never initializes DDS."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from deploy.common.go2_policy import PolicyObservationHistory, build_observation, projected_gravity, action_to_target
from deploy.common.him_qualification import sha256
from deploy.deploy_real.deploy_real_go2 import Go2RealConfig, Go2Controller, quaternion_to_roll_pitch

ROOT = Path(LEGGED_GYM_ROOT_DIR)


def audit(config, path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    active = [row for row in rows if row['mode'] == 'policy']
    if not active:
        raise ValueError('Log contains no policy samples')
    policy_hash = sha256(config.policy_path)
    if any(row['feedback'].get('policy_sha256') != policy_hash or
           row['feedback'].get('policy_config_sha256') != config.source_sha256 for row in active):
        raise ValueError('Recorded model/configuration does not match replay inputs')
    policy = torch.jit.load(config.policy_path, map_location='cpu').eval()
    history = PolicyObservationHistory(config.data)
    processor = Go2Controller(config, policy=policy, action_step_mode='limit')
    previous_raw = np.zeros(12, dtype=np.float32)
    inference = 0
    counts, raw_errors, filtered_errors, target_errors, replay_times = [], [], [], [], []
    observations, raw_actions = [], []
    for row in active:
        f = row['feedback']
        count = f['policy_inference_count']
        if count == inference:
            continue
        if count != inference + 1:
            raise ValueError(f'Inference missing: {inference} -> {count}; cannot reconstruct history')
        q = np.asarray(row['joint_pos_sdk'])[config.policy_to_sdk]
        dq = np.asarray(row['joint_vel_sdk'])[config.policy_to_sdk]
        frame = build_observation(row['angular_velocity'], projected_gravity(row['quaternion_wxyz']),
                                  row['velocity_command'], q, dq, previous_raw, config.data)
        obs = history.append(frame)
        with torch.inference_mode():
            predicted = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().reshape(-1)
        recorded = np.asarray(f['policy_action_processing']['raw_action'], dtype=np.float32)
        processor.action_step_limit = f['policy_action_processing']['step_limit']
        applied = processor.process_policy_action(recorded)
        processor.last_action = applied.copy()
        target = action_to_target(applied, config.data)
        raw_errors.append(float(np.max(np.abs(predicted-recorded))))
        filtered_errors.append(float(np.max(np.abs(applied-f['policy_action_processing']['applied_action']))))
        target_errors.append(float(np.max(np.abs(target-f['policy_target_processing']['raw_target_policy']))))
        counts.append(count);replay_times.append(f['test_elapsed'])
        observations.append(obs);raw_actions.append(recorded)
        previous_raw = np.clip(recorded, -config.data['actions']['clip'], config.data['actions']['clip'])
        inference = count
    timestamps = np.asarray([r['time'] for r in active])
    state_ages = np.asarray([r['time']-r['state_time'] for r in active])
    tau = np.asarray([r['feedback']['joint_tau_est_sdk'] for r in active])
    q = np.asarray([r['joint_pos_sdk'] for r in active]);dq = np.asarray([r['joint_vel_sdk'] for r in active])
    target = np.asarray([r['target_pos_sdk'] for r in active])
    pd = np.asarray([r['kp_sdk'] for r in active])*(target-q)-np.asarray([r['kd_sdk'] for r in active])*dq
    def encoded_command(q,kp,kd):
        return np.asarray([*q,*kp,*kd],dtype=np.float32).tobytes()
    sent_commands = {encoded_command(r['target_pos_sdk'],r['kp_sdk'],r['kd_sdk']) for r in rows}
    echo_checked,echo_matches,echo_bad_crc = 0,0,0
    for row in active:
        echo = row['feedback'].get('observed_lowcmd')
        if echo:
            echo_checked += 1
            echo_bad_crc += not echo.get('crc_valid')
            echo_matches += encoded_command(echo['target_pos_sdk'],echo['kp_sdk'],echo['kd_sdk']) in sent_commands
    # Independent kinematic checks: encoder derivatives versus reported dq;
    # quaternion body increments versus gyroscope. Neither uses the policy.
    unique = []
    for row in active:
        if not unique or row['feedback']['state_tick'] != unique[-1]['feedback']['state_tick']:
            unique.append(row)
    ticks=np.asarray([r['feedback']['state_tick'] for r in unique],dtype=np.int64)
    dt=np.diff(ticks)*.001
    valid=(dt>0)&(dt<=.02)
    uq=np.asarray([r['joint_pos_sdk'] for r in unique])
    udq=np.asarray([r['joint_vel_sdk'] for r in unique])
    derivative=np.diff(uq,axis=0)[valid]/dt[valid,None]
    average_dq=((udq[:-1]+udq[1:])*.5)[valid]
    def cosine_columns(a,b):
        return (np.sum(a*b,axis=0)/np.maximum(np.linalg.norm(a,axis=0)*np.linalg.norm(b,axis=0),1e-12)).tolist()
    quat=np.asarray([r['quaternion_wxyz'] for r in unique],dtype=np.float64);quat/=np.linalg.norm(quat,axis=1,keepdims=True)
    w0=quat[:-1,:1];v0=quat[:-1,1:];w1=quat[1:,:1];v1=quat[1:,1:]
    relative_vec=w0*v1-w1*v0-np.cross(v0,v1)
    relative_w=(w0*w1+np.sum(v0*v1,axis=1,keepdims=True))
    relative_vec*=np.where(relative_w<0,-1.,1.)
    quat_omega=2.*relative_vec[valid]/dt[valid,None]
    gyro=np.asarray([r['angular_velocity'] for r in unique])
    average_gyro=((gyro[:-1]+gyro[1:])*.5)[valid]
    fault = next((r for r in rows if r.get('reason')), None)
    report = {'offline_only': True, 'dds_initialized': False, 'model_unchanged': True,
              'policy_sha256': policy_hash, 'config_sha256': config.source_sha256,
              'log_sha256': sha256(path), 'log': str(path), 'inferences_replayed': inference,
              'max_raw_action_error': max(raw_errors), 'max_filtered_action_error': max(filtered_errors),
              'max_decoded_target_error_rad': max(target_errors),
              'replay_matched': bool(max(raw_errors)<1e-5 and max(filtered_errors)<1e-6 and max(target_errors)<1e-6),
              'policy_interval_ms': {'min':float(np.min(np.diff(replay_times))*1000),
                                     'max':float(np.max(np.diff(replay_times))*1000)},
              'publication_interval_ms': {'max':float(np.max(np.diff(timestamps))*1000),
                                          'p99':float(np.percentile(np.diff(timestamps),99)*1000)},
              'max_received_state_age_ms': float(state_ages.max()*1000),
              'max_quaternion_norm_error':float(np.max(np.abs(np.linalg.norm([r['quaternion_wxyz'] for r in active],axis=1)-1))),
              'valid_crc_echo_samples':echo_checked-echo_bad_crc,'echo_command_matches':echo_matches,
              'echo_samples':echo_checked,'bad_crc_echo_samples':echo_bad_crc,
              'encoder_derivative_vs_dq_cosine_by_sdk_joint':cosine_columns(derivative,average_dq),
              'quaternion_body_increment_vs_gyro_cosine':cosine_columns(quat_omega,average_gyro),
              'sensor_check_scope':'Tick treated as milliseconds; filtering and latency affect these consistency metrics. Physical mounting and encoder zeros remain uncalibrated.',
              'max_abs_roll_pitch_rad':np.max(np.abs([quaternion_to_roll_pitch(r['quaternion_wxyz']) for r in active]),axis=0).tolist(),
              'max_feedback_torque_nm_by_sdk_joint':np.max(np.abs(tau),axis=0).tolist(),
              'max_pd_torque_nm_by_sdk_joint':np.max(np.abs(pd),axis=0).tolist(),
              'fault_reason': fault.get('reason') if fault else None,
              'scope': 'Replay validates recorded software arithmetic, not physical joint identity, encoder calibration, IMU mounting, or actuator response.'}
    arrays = dict(observations=np.asarray(observations),raw_actions=np.asarray(raw_actions),
                  inference_elapsed=np.asarray(replay_times),publication_elapsed=np.asarray([r['feedback']['test_elapsed'] for r in active]),
                  q_sdk=q,dq_sdk=dq,target_sdk=target,pd_torque_sdk=pd,feedback_torque_sdk=tau)
    return report, arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'deploy/deploy_real/configs/go2_him_fresh.yaml')
    parser.add_argument('--log',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    report,arrays=audit(Go2RealConfig.load(args.config,str(ROOT)),args.log)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'),**arrays)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['replay_matched'] else 1


if __name__=='__main__':raise SystemExit(main())
