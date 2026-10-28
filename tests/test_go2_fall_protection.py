"""Fall stop stays active with trajectory shaping disabled; no DDS access."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from deploy.deploy_real.deploy_real_go2 import Go2Controller, Go2DDSTransport, Go2Mode, Go2RealConfig, Go2State
from deploy.deploy_real.go2_commission import PolicyCheck, VerifiedTransport
from deploy.deploy_real.go2_watchdog import CommandGate

ROOT = Path(__file__).resolve().parents[1]


class FallProtectionTest(unittest.TestCase):
    def config(self):
        cfg = Go2RealConfig.load(ROOT/'deploy/deploy_real/configs/go2_him_reward_restore_v2.yaml', str(ROOT))
        cfg.data['software_motion_limits'] = False
        return cfg

    def state(self, cfg, now, roll=0.):
        state = Go2State(now, np.array([np.cos(roll/2), np.sin(roll/2), 0., 0.]), np.zeros(3),
                         cfg.default_joint_pos[cfg.sdk_to_policy].copy(), np.zeros(12))
        state.feedback = {'joint_tau_est_sdk': [0.]*12, 'state_tick': int(now*500)}
        return state

    def assert_damping(self, command):
        self.assertEqual(command.mode, Go2Mode.DAMPING)
        np.testing.assert_array_equal(command.kp_sdk, np.zeros(12))
        self.assertTrue(np.all(command.kd_sdk > 0))
        self.assertIn('摔倒保护', command.reason)

    def test_controller_latches_fall_stop_with_motion_limits_disabled(self):
        cfg = self.config()
        ctrl = Go2Controller(cfg, policy=lambda obs: torch.ones((1, 12)))
        ctrl.fsm.mode = Go2Mode.POLICY
        self.assertEqual(ctrl.step(1., self.state(cfg, 1., .34)).mode, Go2Mode.POLICY)
        self.assert_damping(ctrl.step(1.002, self.state(cfg, 1.002, .36)))
        self.assert_damping(ctrl.step(1.004, self.state(cfg, 1.004)))

    def test_trial_stops_in_entry_policy_and_c_exit_without_resuming(self):
        for phase in ('entry', 'policy', 'exit'):
            with self.subTest(phase=phase):
                cfg = self.config()
                trial = PolicyCheck(cfg, None, policy=lambda obs: torch.zeros((1, 12)), forward=True, forward_speed=.4,
                                    keyboard=True, keyboard_axis='all', continuous=True, turn_forward_speed=.3)
                trial.arm(1., self.state(cfg, 1.))
                now = 1.002 if phase == 'entry' else 8.
                if phase == 'exit':
                    trial.hold_samples = 100
                    trial.hold_first, trial.hold_last = 6., 7.
                    trial.output(8., self.state(cfg, 8.))
                    trial.feed_key('c', 8.02)
                    self.assertIsNotNone(trial.normal_exit)
                    now = 10.12
                self.assert_damping(trial.output(now, self.state(cfg, now, .36)))
                self.assert_damping(trial.output(now+.002, self.state(cfg, now+.002)))

    def test_lowering_checks_attitude_before_generating_position_trajectory(self):
        cfg = self.config()
        trial = PolicyCheck(cfg, None, policy=lambda obs: torch.zeros((1, 12)), forward=True, forward_speed=.4,
                            keyboard=True, keyboard_axis='all', continuous=True, turn_forward_speed=.3)
        trial.arm(1., self.state(cfg, 1.))
        trial.hold_samples = 100
        trial.hold_first, trial.hold_last = 6., 7.
        trial.output(8., self.state(cfg, 8.))
        trial.feed_key('c', 8.02)
        trial.last_time = 10.118
        with self.assertRaisesRegex(ValueError, '摔倒保护'):
            trial.controlled_lowering.command(trial, self.state(cfg, 10.12, .36), 10.12)

    def test_independent_watchdog_latches_fall_stop(self):
        cfg = self.config()
        gate = CommandGate(cfg, 10.)
        q = cfg.default_joint_pos[cfg.sdk_to_policy]
        gate.accept({'time': 1., 'mode': 'policy', 'target': q.tolist(), 'kp': [20.]*12, 'kd': [.5]*12},
                    1., self.state(cfg, 1.))
        self.assert_damping(gate.output(1.002, self.state(cfg, 1.002, .36)))
        self.assert_damping(gate.output(1.004, self.state(cfg, 1.004)))

    def test_blocked_candidate_rejected_before_publisher_or_factory_handoff(self):
        cfg = self.config()
        cfg.data['hardware_deployment_blocked'] = True
        cfg.data['hardware_deployment_block_reason'] = '翻车候选已封停'
        with self.assertRaisesRegex(ValueError, '已封停'):
            Go2DDSTransport(cfg, publish=True)
        with self.assertRaisesRegex(ValueError, '已封停'):
            VerifiedTransport.prepare_factory_handoff(SimpleNamespace(_commission_config=cfg))


if __name__ == '__main__':
    unittest.main()
