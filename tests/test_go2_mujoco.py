import os
import unittest

import mujoco
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(ROOT, "resources", "robots", "go2", "mjcf", "scene.xml")
JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
DEFAULT_POS = np.asarray([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])


class Go2MujocoModelTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)

    def test_model_contract(self):
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu), (19, 18, 12))
        qpos_ids = [int(self.model.joint(name).qposadr[0]) for name in JOINT_NAMES]
        self.assertEqual(qpos_ids, list(range(7, 19)))
        actuator_ids = []
        for name in JOINT_NAMES:
            joint_id = int(self.model.joint(name).id)
            actuator_ids.append(int(np.flatnonzero(self.model.actuator_trnid[:, 0] == joint_id)[0]))
        self.assertEqual(actuator_ids, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])

    def test_default_pose_pd_stays_upright(self):
        data = mujoco.MjData(self.model)
        qpos_ids = np.asarray([int(self.model.joint(name).qposadr[0]) for name in JOINT_NAMES])
        qvel_ids = np.asarray([int(self.model.joint(name).dofadr[0]) for name in JOINT_NAMES])
        actuator_ids = []
        for name in JOINT_NAMES:
            joint_id = int(self.model.joint(name).id)
            actuator_ids.append(int(np.flatnonzero(self.model.actuator_trnid[:, 0] == joint_id)[0]))
        actuator_ids = np.asarray(actuator_ids)
        data.qpos[qpos_ids] = DEFAULT_POS
        mujoco.mj_forward(self.model, data)
        torque_limits = np.asarray([23.7, 23.7, 45.43] * 4)
        for _ in range(2500):
            torque = 20.0 * (DEFAULT_POS - data.qpos[qpos_ids]) - 0.5 * data.qvel[qvel_ids]
            data.ctrl[actuator_ids] = np.clip(torque, -torque_limits, torque_limits)
            mujoco.mj_step(self.model, data)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertGreater(data.qpos[2], 0.20)


if __name__ == "__main__":
    unittest.main()
