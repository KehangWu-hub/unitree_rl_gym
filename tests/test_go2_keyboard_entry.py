import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deploy.deploy_real import go2_keyboard as entry
from deploy.deploy_real import go2_commission as commission


class KeyboardEntryTest(unittest.TestCase):
    def test_fixed_profiles_preserve_existing_limits(self):
        with patch('sys.stdin.isatty', return_value=True):
            turn = commission.parse_args(entry.commission_arguments('walk-turn', 'eno1', Path('zero.jsonl')))
            zero = commission.parse_args(entry.commission_arguments('zero', 'eno1', Path('stand.jsonl')))
        self.assertEqual(turn.duration, 29)
        self.assertEqual(turn.turn_forward_speed, .3)
        self.assertEqual(turn.yaw_speed, .3)
        self.assertEqual(turn.keyboard_axis, 'yaw')
        self.assertTrue(turn.keyboard)
        self.assertEqual(zero.duration, 16)
        self.assertFalse(zero.keyboard)
        self.assertTrue(zero.start_standing)

    def test_backward_and_lateral_profiles_are_isolated_and_bounded(self):
        for mode in ('backward', 'lateral'):
            with self.subTest(mode=mode), patch('sys.stdin.isatty', return_value=True):
                args = commission.parse_args(entry.commission_arguments(mode, 'eno1', Path('zero.jsonl')))
            self.assertEqual(args.keyboard_axis, mode)
            self.assertEqual(args.duration, 29)
            self.assertEqual(args.turn_forward_speed, 0.)
            self.assertEqual(args.publish_hz, 500)
            self.assertTrue(args.start_standing)
            self.assertFalse(args.restore_normal)

    def test_changed_model_or_configuration_cannot_use_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)/'model'; config = Path(directory)/'config'
            model.write_bytes(b'model'); config.write_bytes(b'config')
            cfg = SimpleNamespace(policy_path=str(model))
            manifest = {'policy_sha256': entry.sha256(model), 'config_sha256': entry.sha256(config)}
            entry.verify_baseline(cfg, manifest, config)
            model.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, '策略'): entry.verify_baseline(cfg, manifest, config)
            model.write_bytes(b'model'); config.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, '配置'): entry.verify_baseline(cfg, manifest, config)

    def test_unknown_readiness_requires_matching_model_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'zero.jsonl'
            manifest = {'readiness': {'zero': {'sha256': 'historical-pinned-hash'}},
                        'policy_sha256': 'model', 'config_sha256': 'config'}
            path.write_text(json.dumps({'mode':'policy','feedback':{}}))
            with patch.object(commission, 'PolicyCheck') as controller:
                with self.assertRaisesRegex(ValueError, '模型/配置标记'):
                    entry.validate_readiness(path, 'walk-turn', None, manifest)
                controller.assert_not_called()
                path.write_text(json.dumps({'mode':'policy','feedback':{
                    'policy_sha256':'model','policy_config_sha256':'config'}}))
                entry.validate_readiness(path, 'walk-turn', None, manifest)
                controller.assert_called_once()

    def test_new_stand_record_is_accepted_only_with_matching_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'stand.jsonl'
            manifest = {'readiness':{'stand':{'sha256':'old-record'}}, 'config_sha256':'config'}
            with patch.object(commission,'PolicyCheck') as controller:
                path.write_text(json.dumps({'mode':'fix_stand','feedback':{'control_config_sha256':'wrong'}}))
                with self.assertRaisesRegex(ValueError,'模型/配置标记'):
                    entry.validate_readiness(path,'zero',None,manifest)
                controller.assert_not_called()
                path.write_text(json.dumps({'mode':'fix_stand','feedback':{'control_config_sha256':'config'}}))
                entry.validate_readiness(path,'zero',None,manifest)
                controller.assert_called_once()

    def test_offline_and_cancelled_start_never_run_controller(self):
        trial = SimpleNamespace(policy_pd_limit=14., torque_limit=16., duration=29., keyboard_window_end=20., keyboard_budget=4., lower_begin=22.,
                                controller=SimpleNamespace(validate_policy=lambda: None))
        manifest = {'name':'test','readiness':{'zero':{'path':'zero.jsonl'}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'baseline.json'; path.write_text(json.dumps(manifest))
            with patch.object(entry,'BASELINE',path), patch.object(entry.Go2RealConfig,'load'), \
                 patch.object(entry,'verify_baseline'), patch.object(entry,'validate_readiness',return_value=trial), \
                 patch.object(commission,'run') as run, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entry.main([]), 0)
                with patch('sys.stdin.isatty',return_value=True), patch('builtins.input',return_value='cancel'):
                    self.assertEqual(entry.main(['--run']), 0)
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
