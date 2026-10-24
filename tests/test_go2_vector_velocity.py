"""Simulation feedback bounds and contact handling must remain explicit."""
import unittest

import numpy as np
import mujoco
from pathlib import Path

from deploy.common.go2_velocity_control import VectorVelocityPI, stance_velocity_damping


class VectorVelocityTest(unittest.TestCase):
    def test_stalled_reverse_builds_compensation_without_changing_target(self):
        controller = VectorVelocityPI([[-1, 1]] * 3)
        for _ in range(500):
            command = controller.update([-.3, 0, 0], [0, 0, 0], .02)
        self.assertLess(command[0], -.3)
        self.assertGreaterEqual(command[0], -1)
        np.testing.assert_allclose(controller.reference, [-.3, 0, 0])
        integral = controller.integral.copy()
        for _ in range(500):
            controller.update([-.3, 0, 0], [0, 0, 0], .02)
        np.testing.assert_allclose(controller.integral, integral)

    def test_stop_and_sign_reversal_keep_slew_and_correct_uncommanded_axes(self):
        controller = VectorVelocityPI([[-1, 1]] * 3)
        previous = np.zeros(3)
        for target in ([.3, 0, 0], [-.3, 0, 0], [0, 0, 0]):
            for _ in range(100):
                command = controller.update(target, [.1, .03, -.02], .02)
                self.assertTrue(np.all(np.abs(command - previous) <= .020000001))
                previous = command
        self.assertLess(command[1], 0)
        self.assertGreater(command[2], 0)
        np.testing.assert_allclose(controller.reference, 0)

    def test_nonfinite_feedback_rejected_before_state_change(self):
        controller = VectorVelocityPI([[-1, 1]] * 3)
        with self.assertRaises(ValueError):
            controller.update([.3, 0, 0], [np.nan, 0, 0], .02)
        np.testing.assert_array_equal(controller.integral, 0)




    def test_oracle_stance_residual_is_zero_without_ground_contact(self):
        root = Path(__file__).resolve().parents[1]
        model = mujoco.MjModel.from_xml_path(str(root / 'resources/robots/go2/mjcf/scene_flat.xml'))
        data = mujoco.MjData(model)
        data.qpos[2] = 2.
        mujoco.mj_forward(model, data)
        correction = stance_velocity_damping(model, data, np.arange(6, 18), [2, 2, 2], [0, 0, 0], 10., 20., 2.)
        np.testing.assert_array_equal(correction, np.zeros(12))


if __name__ == '__main__':
    unittest.main()
