"""Flat-ground MuJoCo gait measurements shared by offline evaluation tools.

No DDS access. Contact and flight thresholds are analysis criteria only.
"""
import json

import mujoco
import numpy as np

from deploy.deploy_mujoco.validate_go2_him import simulate

LEGS = ('FR', 'FL', 'RR', 'RL')
COMMANDS = {'zero': [0, 0, 0], 'w': [.3, 0, 0], 's': [-.3, 0, 0],
            'a': [0, .3, 0], 'd': [0, -.3, 0], 'q': [.3, 0, .3], 'e': [.3, 0, -.3]}


def measure(config, key, output):
    rows = []
    foot_ids = None

    def observe(model, data, t, command):
        nonlocal foot_ids
        if not 8. <= t < 15.:
            return
        if foot_ids is None:
            foot_ids = [model.geom(name).id for name in LEGS]
        force, slip = np.zeros(4), np.zeros(4)
        clearance = [data.geom_xpos[gid, 2] - model.geom_size[gid, 0] for gid in foot_ids]
        for j in range(data.ncon):
            contact = data.contact[j]
            for i, gid in enumerate(foot_ids):
                other = contact.geom2 if contact.geom1 == gid else contact.geom1 if contact.geom2 == gid else -1
                if other >= 0 and model.geom_bodyid[other] == 0:
                    f = np.zeros(6)
                    mujoco.mj_contactForce(model, data, j, f)
                    force[i] += abs(f[0])
                    jac, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                    mujoco.mj_jac(model, data, jac, jacr, contact.pos, int(model.geom_bodyid[gid]))
                    slip[i] = max(slip[i], float(np.linalg.norm((jac @ data.qvel)[:2])))
        rows.append([t, *force, *clearance, *slip])

    trace = []
    flow, body = simulate(config, key, COMMANDS[key], protected=True, observer=observe, joint_trace=trace)
    feet = np.asarray(rows).reshape(-1, 13)
    steady = feet[feet[:, 0] >= 10.]
    result = {'flow': flow, 'window_seconds': [10., 15.],
              'complete_window': bool(len(steady) and steady[-1, 0] >= 14.998 - 1e-6)}
    if len(steady):
        force, height, slip = steady[:, 1:5], steady[:, 5:9], steady[:, 9:13]
        contact = force > 5.
        flights = {name: [] for name in LEGS}
        for i, name in enumerate(LEGS):
            start = None
            for j in range(len(steady)):
                # Exclude a flight already in progress at the window boundary.
                if j and contact[j-1, i] and not contact[j, i]:
                    start = j
                if contact[j, i] and start is not None:
                    if steady[j, 0] - steady[start, 0] >= .04:
                        flights[name].append(float(height[start:j, i].max()))
                    start = None
        result.update(flight_peak_clearance_m=flights,
                      median_flight_peak_clearance_mm=[float(np.median(flights[n]))*1000 if flights[n] else None for n in LEGS],
                      contact_slip_distance_cm=(slip*contact).sum(axis=0).__mul__(.002*100).tolist(),
                      contact_fraction=contact.mean(axis=0).tolist(),
                      peak_normal_force_N=force.max(axis=0).tolist())
    joints = np.asarray(trace)
    selected = joints[(joints[:, 0] >= 10.) & (joints[:, 0] < 15.)][::10]
    if len(selected) > 2:
        # Trace layout: t, q12, dq12, target12, kp12, kd12, applied_tau12.
        target = selected[:, 25:37]
        result['target_step_rms_rad'] = float(np.sqrt(np.mean(np.diff(target, axis=0)**2)))
        result['target_step_max_rad'] = float(np.abs(np.diff(target, axis=0)).max())
        result['target_second_difference_rms_rad'] = float(np.sqrt(np.mean(np.diff(target, n=2, axis=0)**2)))
    np.savez_compressed(output.with_suffix('.npz'), feet=feet, body=body, joints=joints)
    output.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result
