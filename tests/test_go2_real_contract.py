import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stderr
from io import StringIO

import numpy as np

from deploy.deploy_real.deploy_real_go2 import Go2Command, Go2DDSTransport, Go2Mode, Go2RealConfig, parse_args, run


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_CONFIG = os.path.join(ROOT, "deploy", "deploy_real", "configs", "go2.yaml")


class Go2RealContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Go2RealConfig.load(RUNTIME_CONFIG, ROOT)

    def test_exported_contract_is_the_only_policy_source(self):
        self.assertEqual(self.cfg.data["policy"]["num_observations"], 45)
        self.assertEqual(self.cfg.data["policy"]["num_actions"], 12)
        self.assertAlmostEqual(self.cfg.control_dt, 0.02, places=6)
        np.testing.assert_allclose(self.cfg.stiffness, 20.0)
        np.testing.assert_allclose(self.cfg.damping, 0.5)

    def test_joint_mapping_round_trip(self):
        policy = np.arange(12)
        sdk = np.empty(12, dtype=np.int64)
        sdk[self.cfg.policy_to_sdk] = policy
        np.testing.assert_array_equal(sdk[self.cfg.policy_to_sdk], policy)
        np.testing.assert_array_equal(sdk, policy[self.cfg.sdk_to_policy])

    def test_command_is_conservatively_clipped(self):
        np.testing.assert_allclose(self.cfg.clip_command([1, -1, 2]), [0.5, -0.3, 0.5])

    def test_go2_lowcmd_uses_official_servo_mode(self):
        motors = [SimpleNamespace() for _ in range(20)]
        transport = Go2DDSTransport.__new__(Go2DDSTransport)
        transport._low_cmd = SimpleNamespace(head=[0, 0], motor_cmd=motors)
        transport._init_low_cmd()
        self.assertTrue(all(motor.mode == 0x01 for motor in motors))

        sent = []
        transport._publish = True
        transport._crc = SimpleNamespace(Crc=lambda _cmd: 123)
        transport._publisher = SimpleNamespace(Write=sent.append)
        command = Go2Command(Go2Mode.PASSIVE, np.zeros(12), np.zeros(12), np.ones(12))
        transport.send(command)
        self.assertEqual(sent, [transport._low_cmd])
        self.assertTrue(all(motor.mode == 0x01 for motor in motors[:12]))

    def test_shadow_policy_cannot_publish(self):
        with patch("sys.argv", ["go2", "eno1", "--shadow-policy"]), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parse_args()
        with patch("sys.argv", ["go2", "eno1", "--read-only", "--shadow-policy"]):
            args = parse_args()
        self.assertTrue(args.read_only and args.shadow_policy)

    def test_keyboard_only_real_control_fails_before_dds(self):
        with patch("sys.argv", ["go2", "eno1"]), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parse_args()
        with self.assertRaisesRegex(RuntimeError, "仅允许--read-only"):
            run(SimpleNamespace(check=False, network="eno1", read_only=False))


if __name__ == "__main__":
    unittest.main()
