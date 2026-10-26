"""HIM sim2sim and real-controller flow audit. No DDS initialization or robot access."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from deploy.common.go2_policy import PolicyObservationHistory, build_observation, action_to_target, projected_gravity
from deploy.common.him_qualification import sha256, current_sources
from deploy.deploy_mujoco.deploy_go2 import joint_addresses, apply_model_overrides, load_runtime_config
from deploy.deploy_real.deploy_real_go2 import Go2RealConfig, Go2State, Go2Mode
from deploy.deploy_real.go2_commission import PolicyCheck

ROOT = Path(LEGGED_GYM_ROOT_DIR)


class SimulationPolicyCheck(PolicyCheck):
    """Only bypass physical predecessor evidence inside this non-DDS simulator.

    The production state machine, limits, command schedule, and exit are unchanged.
    Simulation outputs are explicitly tagged and cannot serve as physical readiness.
    """
    def _validate_readiness(self, readiness_log, config, forward):
        if readiness_log is not None:
            raise ValueError('Simulation must not consume physical readiness')


def simulate(config, label, command, protected=False, exit_key='c', original_dynamics=False, stage='keyboard'):
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'resources/robots/go2/mjcf/scene_flat.xml'))
    model.opt.timestep = .001  # Two physics/PD substeps per 500Hz command publication.
    runtime = load_runtime_config(ROOT / 'deploy/deploy_mujoco/configs/go2_him.yaml')
    if not original_dynamics:
        apply_model_overrides(model, runtime.get('model_overrides', {}))
    data = mujoco.MjData(model)
    pos, vel, motors = joint_addresses(model, config.data['policy']['sdk_joint_names'])
    mapping = config.policy_to_sdk
    nominal = config.default_joint_pos[config.sdk_to_policy]
    data.qpos[pos], data.qpos[2] = nominal, .33
    mujoco.mj_forward(model, data)
    for _ in range(3000):
        data.ctrl[motors] = np.clip(40 * (nominal-data.qpos[pos])-data.qvel[vel], -8, 8)
        mujoco.mj_step(model, data)
    policy = torch.jit.load(config.policy_path, map_location='cpu').eval()
    history = PolicyObservationHistory(config.data)
    action = np.zeros(12, dtype=np.float32)
    target = nominal.copy()
    kp, kd = config.stiffness[config.sdk_to_policy], config.damping[config.sdk_to_policy]
    torque_limits = config.torque_limits[config.sdk_to_policy]
    torque_limits = np.minimum(torque_limits, model.actuator_ctrlrange[motors, 1])
    tau = data.ctrl[motors].copy()
    base_id = model.body('base_link').id
    trial = None
    if protected and stage == 'keyboard':
        trial = SimulationPolicyCheck(config, None, forward=True, forward_speed=.4,
                keyboard=True, keyboard_axis='all', turn_forward_speed=.3,
                continuous=True, walk_speed=.3, left_speed=.3, yaw_speed=.3,
                action_step_mode='limit', speed_profile='legacy')
    elif protected:
        trial = SimulationPolicyCheck(config, None, forward=stage == 'forward', forward_speed=.3,
                                       action_step_mode='limit')

    def state(now, tick):
        s = Go2State(now, data.sensor('imu_quat').data.copy(), data.sensor('imu_gyro').data.copy(),
                     data.qpos[pos].copy(), data.qvel[vel].copy())
        s.feedback = {'joint_tau_est_sdk': tau.tolist(), 'state_tick': tick}
        return s

    if trial:
        trial.arm(1., state(1., 0))
    samples, fault, fell = [], '', False
    exit_state = None
    accepted_keys = set()
    max_tau, action_limited, max_action_delta = 0., 0, 0.
    stop_at, exit_at = (15., 19.) if protected else (10., 13.)
    if protected and stage != 'keyboard':
        stop_at, exit_at = (10., 12.) if stage == 'forward' else (9., 9.)
    horizon = 46. if protected else 13.
    for tick in range(int(horizon / .002)):
        t = tick * .002
        now = 1.+t
        if protected:
            if stage == 'keyboard' and 9. <= t < stop_at and tick % 20 == 0 and label != 'zero':
                key = label
                if label == 'switch':
                    key = ('w','a','q','s','d','e')[min(5, int(t-9.))]
                trial.feed_key(key, now)
                if trial.keyboard_event and trial.keyboard_event['accepted']:
                    accepted_keys.add(key)
            if stage == 'keyboard' and tick == round(stop_at/.002):
                trial.feed_key('x', now)
            if stage == 'keyboard' and tick == round(exit_at/.002):
                if exit_key == 'z':
                    trial.stop(now, 'simulation Z emergency stop')
                else:
                    trial.feed_key('c', now)
            out = trial.output(now, state(now, tick))
            target, kp, kd = out.target_pos_sdk, out.kp_sdk, out.kd_sdk
            current_command = trial.controller.command.copy()
            if out.mode is Go2Mode.DAMPING:
                fault = out.reason
                exit_state = {'joint_pos_sdk': data.qpos[pos].tolist(), 'joint_vel_sdk': data.qvel[vel].tolist(),
                              'quaternion_wxyz': data.sensor('imu_quat').data.tolist(),
                              'max_crouch_error':float(np.max(np.abs(data.qpos[pos]-trial.lowered))),
                              'height':float(data.qpos[2])}
                # Exercise the same final one-second damping command in physics.
                for _ in range(1000):
                    data.ctrl[motors] = np.clip(-float(config.data['fsm']['damping_kd'])*data.qvel[vel],-torque_limits,torque_limits)
                    mujoco.mj_step(model,data)
                break
            processing = trial.controller.last_action_processing
            if tick % 10 == 0 and processing:
                action_limited += bool(processing['limited_joints'])
                max_action_delta = max(max_action_delta, processing['max_requested_delta'])
        elif tick % 10 == 0:
            current_command = np.asarray(command if t < stop_at else [0.,0.,0.], dtype=np.float32)
            frame = build_observation(data.sensor('imu_gyro').data, projected_gravity(data.sensor('imu_quat').data),
                    current_command, data.qpos[pos][mapping], data.qvel[vel][mapping], action, config.data)
            obs = history.append(frame)
            with torch.inference_mode():
                action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().reshape(-1)
            policy_target = action_to_target(action, config.data)
            action = np.clip(action,-config.data['actions']['clip'],config.data['actions']['clip'])
            target = policy_target[config.sdk_to_policy]
        if tick % 10 == 0:
            gravity = projected_gravity(data.sensor('imu_quat').data)
            tilt = float(np.arccos(np.clip(-gravity[2],-1.,1.)))
            rotation = data.xmat[base_id].reshape(3,3)
            velocity = rotation.T @ data.qvel[:3]
            samples.append([t, *current_command, *velocity[:2], float(data.sensor('imu_gyro').data[2]),
                            float(data.qpos[2]), tilt])
            if data.qpos[2] < .12 or tilt > .8 or not np.isfinite(data.qpos).all():
                # C lowering intentionally goes below standing height.
                if not protected or trial.normal_exit is None:
                    fell = True
                    break
        tau = kp * (target-data.qpos[pos])-kd*data.qvel[vel]
        max_tau = max(max_tau, float(np.max(np.abs(tau))))
        data.ctrl[motors] = np.clip(tau,-torque_limits,torque_limits)
        mujoco.mj_step(model,data)
        tau = kp * (target-data.qpos[pos])-kd*data.qvel[vel]
        data.ctrl[motors] = np.clip(tau,-torque_limits,torque_limits)
        mujoco.mj_step(model,data)
    rows = np.asarray(samples)
    moving_begin = (8.3 if stage == 'forward' else 7.5) if protected and stage != 'keyboard' else 10. if protected else 2.
    moving = rows[(rows[:,0] >= moving_begin) & (rows[:,0] < stop_at)]
    errors = moving[:,4:7]-moving[:,1:4] if len(moving) else np.full((1,3),np.inf)
    means = np.mean(moving[:,4:7],axis=0) if len(moving) else np.zeros(3)
    requested = np.mean(moving[:,1:4],axis=0) if len(moving) else np.zeros(3)
    active = np.abs(requested) > .05
    ratio = means[active]/requested[active]
    stopped_end = exit_at if stage == 'keyboard' else stop_at+6.
    stopped = rows[(rows[:,0] >= stop_at+2.) & (rows[:,0] < stopped_end)]
    residual = float(np.mean(np.linalg.norm(stopped[:,4:6],axis=1))) if len(stopped) else float('inf')
    residual_yaw = float(np.mean(np.abs(stopped[:,6]))) if len(stopped) else float('inf')
    zero_ok = bool(len(stopped) and np.max(np.abs(stopped[:,1:4])) < 1e-6)
    rmse = np.sqrt(np.mean(errors**2,axis=0))
    tracking = bool(len(moving) and (label == 'switch' or np.all(ratio >= .5)) and np.all(rmse <= [.20,.20,.25]))
    flow = not protected or (fault == ('simulation Z emergency stop' if exit_key == 'z' else '测试时长结束') and t >= exit_at)
    passed = bool(not fell and flow and tracking and zero_ok and residual < .08 and residual_yaw < .15)
    return {'label':label, 'stage':stage, 'protected':protected, 'exit_key':exit_key,
            'physics_dt':float(model.opt.timestep), 'publish_dt':.002,
            'dynamics':'original_mjcf' if original_dynamics else 'training_overrides',
            'passed':passed, 'flow_passed':bool(flow), 'tracking_passed':tracking, 'fell':fell,
            'reason':fault, 'exit_state':exit_state, 'accepted_keys':sorted(accepted_keys), 'elapsed':t, 'mean_command':requested.tolist(), 'mean_velocity':means.tolist(),
            'rmse':rmse.tolist(), 'tracking_ratio':ratio.tolist(), 'zero_command_passed':zero_ok,
            'residual_speed':residual, 'residual_yaw':residual_yaw, 'max_pd_torque':max_tau,
            'limited_inferences':action_limited, 'max_requested_action_delta':max_action_delta}, rows


def qualify(directory, config, config_path):
    """Freeze only complete, matching simulation evidence; never creates field readiness."""
    paths = {stage:directory/name for stage,name in {
        'raw':'final_raw/summary.json', 'protected':'final_protected/summary.json',
        'zero':'zero_stage/summary.json', 'forward':'forward_stage/summary.json'}.items()}
    sources = current_sources()
    policy_hash, config_hash = sha256(config.policy_path), sha256(config_path)
    reports = {stage:json.loads(path.read_text()) for stage,path in paths.items()}
    for stage, data in reports.items():
        if not data['passed'] or data.get('source_sha256') != sources or data['policy_sha256'] != policy_hash or data['config_sha256'] != config_hash:
            raise ValueError(f'Incomplete or stale evidence: {stage}; rerun this stage')
    keys = {'zero','w','s','a','d','q','e'}
    if {r['label'] for r in reports['raw']['cases']} != keys:
        raise ValueError('Raw policy evidence must include zero and six direction commands')
    protected = reports['protected']['cases']
    if {r['label'] for r in protected if r['exit_key'] == 'c'} != keys | {'switch'}:
        raise ValueError('Protected evidence must include all commands and direction switches')
    if not any(r['exit_key'] == 'z' and r['flow_passed'] for r in protected):
        raise ValueError('Missing emergency exit evidence')
    if not any(r['label'] == 'switch' and set(r['accepted_keys']) == keys-{'zero'} for r in protected):
        raise ValueError('Missing six accepted direction switches')
    for stage in ('zero','forward'):
        if len(reports[stage]['cases']) != 1 or reports[stage]['cases'][0]['stage'] != stage:
            raise ValueError(f'Missing standalone commissioning flow: {stage}')
    isaac_path = directory/'isaac.json'
    isaac = json.loads(isaac_path.read_text())
    if isaac['policy_sha256'] != policy_hash or not isaac['finite'] or any(isaac['resets']):
        raise ValueError('Invalid Isaac Gym reference model or episodes')
    if isaac['environment_source_sha256'] != sources['legged_gym/envs/go2_him/go2_him_env.py']:
        raise ValueError('Isaac Gym environment changed since reference measurement')
    rows = []
    for case in reports['raw']['cases']:
        index = isaac['labels'].index(case['label'])
        expected = np.asarray(isaac['mean_velocity'][index])
        actual = np.asarray(case['mean_velocity'])
        delta = actual-expected
        rows.append({'label':case['label'], 'isaac_velocity':expected.tolist(),
                     'mujoco_velocity':actual.tolist(), 'difference':delta.tolist()})
    maximum = np.max(np.abs([r['difference'] for r in rows]),axis=0)
    comparison = {'simulation_only':True, 'passed':bool(np.all(maximum < .05)),
                  'policy_sha256':policy_hash, 'max_absolute_difference':maximum.tolist(),
                  'tolerance':[.05,.05,.05], 'cases':rows,
                  'isaac_reference_sha256':sha256(isaac_path), 'contact_friction':isaac['contact_friction']}
    compare_path = directory/'comparison.json'
    compare_path.write_text(json.dumps(comparison,indent=2)+'\n')
    if not comparison['passed']:
        raise ValueError('Isaac Gym / MuJoCo velocity difference exceeds 0.05')
    paths['comparison'] = compare_path
    paths['isaac'] = isaac_path
    result = {'simulation_only':True, 'passed':True, 'policy_sha256':policy_hash,
              'config_sha256':config_hash, 'source_sha256':sources,
              'evidence':{stage:{'path':str(path.resolve().relative_to(ROOT)), 'sha256':sha256(path)} for stage,path in paths.items()},
              'scope':'Flat ground; 0.30 m/s translations and 0.30 rad/s walking turns, X zero, C lowering, Z stop. Field commissioning required.'}
    output = ROOT/'logs/go2_him/qualification.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'deploy/deploy_real/configs/go2_him.yaml')
    parser.add_argument('--output',type=Path,default=ROOT/'logs/go2_him/phase2_validation')
    parser.add_argument('--policy', type=Path, help='Offline candidate override; does not modify hardware configuration')
    parser.add_argument('--qualify', action='store_true', help='Verify and freeze the complete suite plus Isaac Gym reference')
    parser.add_argument('--stage', choices=('keyboard','zero','forward'), default='keyboard')
    parser.add_argument('--protected', action='store_true')
    parser.add_argument('--original-dynamics',action='store_true')
    parser.add_argument('--keys', nargs='+',choices=('zero','w','s','a','d','q','e','switch'),default=['zero','w','s','a','d','q','e'])
    args = parser.parse_args()
    sources = current_sources()
    torch.set_num_threads(1)
    config=Go2RealConfig.load(args.config,str(ROOT))
    if args.policy is not None:
        config=replace(config,policy_path=str(args.policy.resolve()))
    args.output.mkdir(parents=True,exist_ok=True)
    if args.qualify:
        if args.policy is not None:
            parser.error('--qualify does not accept a policy override')
        return qualify(args.output,config,args.config)
    commands={'zero':[0,0,0],'w':[.3,0,0],'s':[-.3,0,0],'a':[0,.3,0],'d':[0,-.3,0],
              'q':[.3,0,.3],'e':[.3,0,-.3],'switch':[0,0,0]}
    if args.stage != 'keyboard':
        if not args.protected:
            parser.error('--stage zero/forward requires --protected')
        args.keys=['zero' if args.stage == 'zero' else 'w']
    results=[]
    for key in args.keys:
        report,samples=simulate(config,key,commands[key],args.protected,original_dynamics=args.original_dynamics,stage=args.stage)
        results.append(report)
        np.savez_compressed(args.output/(key+'.npz'), samples=samples)
        print(json.dumps(report,ensure_ascii=False),flush=True)
    if args.protected and args.stage == 'keyboard' and 'switch' in args.keys:
        report,samples=simulate(config,'w',commands['w'],True,exit_key='z',original_dynamics=args.original_dynamics)
        results.append(report)
        np.savez_compressed(args.output/'z_exit.npz', samples=samples)
    result={'simulation_only':True,'passed':all(r['passed'] for r in results) and current_sources() == sources,'cases':results,
            'source_sha256': sources,
            'policy_sha256':sha256(config.policy_path),'config_sha256':sha256(args.config)}
    (args.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
