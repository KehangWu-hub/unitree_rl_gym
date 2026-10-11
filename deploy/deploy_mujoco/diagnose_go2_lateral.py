"""离线对照足端接触维度和关节阻力；不初始化DDS，不修改真机参数。"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from deploy.deploy_mujoco.deploy_go2 import joint_addresses
from deploy.deploy_real.deploy_real_go2 import Go2Controller, Go2RealConfig, Go2State, quaternion_to_roll_pitch
from deploy.deploy_real.go2_keyboard import CONFIG, ROOT, sha256


def simulate(config, row, friction, condim, joint_friction, seed_geometry='geom'):
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'resources/robots/go2/mjcf/scene_flat.xml'))
    model.opt.timestep = .002
    model.dof_damping[6:] = 0
    model.dof_armature[6:] = 0
    model.dof_frictionloss[6:] = joint_friction
    model.geom_friction[:, 0] = friction
    feet = [model.geom(name).id for name in ('FR', 'FL', 'RR', 'RL')]
    model.geom_condim[feet] = condim
    pos, vel, actuator = joint_addresses(model, config.data['policy']['sdk_joint_names'])
    data = mujoco.MjData(model)
    data.qpos[2] = 0
    data.qpos[3:7] = row['quaternion_wxyz']
    data.qpos[pos] = row['joint_pos_sdk']
    data.qvel[vel] = row['joint_vel_sdk']
    mujoco.mj_forward(model, data)
    if seed_geometry == 'body':
        # 复核早期脚本采用的足端body中心；实际球心存在2mm局部偏移。
        bodies = [model.body(name + '_foot').id for name in ('FR', 'FL', 'RR', 'RL')]
        data.qpos[2] = .022 - min(data.xpos[bodies, 2])
    else:
        data.qpos[2] = -min(data.geom_xpos[feet, 2] - model.geom_size[feet, 0])
    mujoco.mj_forward(model, data)
    initial = data.qpos[:3].copy()
    controller = Go2Controller(config)
    controller.last_action = ((np.asarray(row['feedback']['policy_target_processing']['raw_target_policy'])
                               - config.default_joint_pos) / config.data['actions']['scale']).astype(np.float32)
    controller.set_velocity_command([0, .3, 0])
    targets, actual, speeds, supports, torques = [], [], [], [], []
    max_tilt = 0.
    min_height = initial[2]
    for tick in range(2500):
        if tick % 10 == 0:
            state = Go2State(data.time, data.sensor('imu_quat').data.copy(),
                             data.sensor('imu_gyro').data.copy(), data.qpos[pos].copy(), data.qvel[vel].copy())
            q, dq = controller._policy_joint_state(state)
            action, target = controller._policy_action(state, q, dq)
            controller.last_action = action
            target_sdk = controller._to_sdk(controller.safety.limit_target(target))
            targets.append(target_sdk.copy())
            actual.append(data.qpos[pos].copy())
            # 机身坐标速度与侧移指令可直接比较。
            rotation = data.xmat[model.body('base_link').id].reshape(3, 3)
            speeds.append(rotation.T @ data.qvel[:3])
            support = np.zeros(4)
            for index in range(data.ncon):
                contact = data.contact[index]
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, force)
                for foot, geom in enumerate(feet):
                    if geom in (contact.geom1, contact.geom2):
                        support[foot] += abs(force[0])
            supports.append(support)
            max_tilt = max(max_tilt, max(abs(x) for x in quaternion_to_roll_pitch(state.quaternion_wxyz)))
            min_height = min(min_height, data.qpos[2])
        torque = np.clip(config.stiffness[config.sdk_to_policy] * (target_sdk - data.qpos[pos])
                         - config.damping[config.sdk_to_policy] * data.qvel[vel], -14., 14.)
        torques.append(torque.copy())
        data.ctrl[actuator] = torque
        mujoco.mj_step(model, data)
    return {
        'sliding_friction': friction, 'foot_contact_dimension': condim,
        'joint_frictionloss_nm': joint_friction, 'initial_height_method': seed_geometry,
        'initial_height_m': float(initial[2]), 'displacement_world_xyz_m': (data.qpos[:3] - initial).tolist(),
        'final_second_mean_body_velocity_m_s': np.mean(speeds[-50:], axis=0).tolist(),
        'final_second_target_range_sdk_rad': np.ptp(targets[-50:], axis=0).tolist(),
        'final_second_actual_range_sdk_rad': np.ptp(actual[-50:], axis=0).tolist(),
        'final_second_support_fraction_gt_2n': np.mean(np.asarray(supports[-50:]) > 2., axis=0).tolist(),
        'final_second_mean_torque_sdk_nm': np.mean(torques[-500:], axis=0).tolist(),
        'max_tilt_rad': float(max_tilt), 'min_height_m': float(min_height),
        'finite': bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, default=ROOT / 'logs/go2_real/commission/go2_real_20261011_154518_1791704718432105569.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/go2_real/commission/lateral_contact_dimension_probe_20261011.json')
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.log.read_text().splitlines() if line.strip()]
    row = [r for r in rows if r.get('mode') == 'policy' and r['velocity_command'][1] > .28][-1]
    config = Go2RealConfig.load(CONFIG, str(ROOT))
    cases = [(mu, dim, loss, 'geom') for mu in (.4, 1., 2., 3.) for dim in (3, 6) for loss in (0., .2)]
    cases += [(3., 6, 0., 'body')]
    results = []
    for mu, dim, loss, geometry in cases:
        result = simulate(config, row, mu, dim, loss, geometry)
        results.append(result)
        print(f'mu={mu:g} dim={dim} joint_loss={loss:g} seed={geometry}: '
              f'final vy={result["final_second_mean_body_velocity_m_s"][1]:.4f}m/s '
              f'target_range={max(result["final_second_target_range_sdk_rad"]):.4f}rad', flush=True)
    report = {'simulation_only': True, 'source_log': str(args.log), 'source_sha256': sha256(args.log),
              'policy_sha256': sha256(config.policy_path), 'config_sha256': sha256(CONFIG),
              'scope': '从历史停滞状态运行5秒纯左移；不含接管/退出，接触参数和初始高度均为假设，不能作为真机验收。',
              'cases': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'报告：{args.output}')


if __name__ == '__main__':
    main()
