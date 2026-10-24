"""Check the estimator's FK/Jacobian and independence from hidden simulator state."""
import unittest
from pathlib import Path

import mujoco
import numpy as np

from deploy.common.go2_velocity_estimator import KinematicVelocityEstimator
from deploy.deploy_real.deploy_real_go2 import Go2RealConfig

ROOT = Path(__file__).resolve().parents[1]


class VelocityEstimatorTest(unittest.TestCase):
    def setUp(self):
        config = Go2RealConfig.load(ROOT / "deploy/deploy_real/configs/go2.yaml", str(ROOT))
        self.q = config.default_joint_pos[config.sdk_to_policy]
        self.estimator = KinematicVelocityEstimator(ROOT / "resources/robots/go2/mjcf/scene_flat.xml",
                         config.data["policy"]["sdk_joint_names"], filter_seconds=0.)

    def test_stationary_estimate_does_not_read_hidden_base_velocity(self):
        estimator = self.estimator
        estimator.data.qvel[:3] = [100, 200, 300]
        estimator.data.qpos[:3] = [100, 200, 300]
        value = estimator.estimate(self.q, np.zeros(12), [1, 0, 0, 0], [0, 0, 0], .02)
        np.testing.assert_array_equal(value, 0.)
        np.testing.assert_array_equal(estimator.data.qvel, 0.)

    def test_joint_jacobian_matches_finite_difference_in_sdk_order(self):
        estimator = self.estimator
        dq = np.linspace(-.2, .2, 12)
        value = estimator.estimate(self.q, dq, [1, 0, 0, 0], [0, 0, 0], .02)
        weights = estimator.weights.copy()
        points, local_points, bodies = [], [], []
        for geom in estimator.feet:
            body = int(estimator.model.geom_bodyid[geom])
            point = estimator.data.geom_xpos[geom].copy()
            point[2] -= estimator.model.geom_size[geom, 0]
            points.append(point)
            local_points.append(estimator.data.xmat[body].reshape(3, 3).T @ (point - estimator.data.xpos[body]))
            bodies.append(body)
        epsilon = 1e-5
        estimator.data.qpos[estimator.qpos_ids] = self.q + epsilon * dq
        mujoco.mj_forward(estimator.model, estimator.data)
        future = np.array([estimator.data.xpos[body] + estimator.data.xmat[body].reshape(3, 3) @ local
                           for body, local in zip(bodies, local_points)])
        numeric = -weights @ ((future - np.array(points)) / epsilon)
        np.testing.assert_allclose(value, numeric, atol=1e-6)

    def test_invalid_observation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid kinematic estimator input"):
            self.estimator.estimate(self.q, np.zeros(12), [0, 0, 0, 0], [0, 0, 0], .02)


if __name__ == "__main__":
    unittest.main()
