"""Continuation restores mappings and config objects without Isaac Gym."""
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from tests.test_go2_him_training import file_module

ROOT = Path(__file__).resolve().parents[1]
MODULE = file_module('him_continuation_tests', ROOT/'legged_gym/scripts/continue_him_training.py')


class ContinuationTest(unittest.TestCase):
    def test_nested_config_keeps_dictionaries_and_replaces_values(self):
        cfg = NS(control=NS(stiffness={'joint': 1.}, damping={'joint': 2.}),
                 init_state=NS(default_joint_angles={'old_joint': 0.}), terrain=NS(curriculum=False))
        saved = {'control': {'stiffness': {'joint': 20.}, 'damping': {'joint': .5}},
                 'init_state': {'default_joint_angles': {'FL_hip_joint': .1}},
                 'terrain': {'curriculum': True, 'num_sub_terrains': 200}}
        before = copy.deepcopy(saved)
        MODULE.overlay(cfg, saved)
        self.assertEqual(cfg.control.stiffness, {'joint': 20.})
        self.assertEqual(cfg.control.damping, {'joint': .5})
        self.assertEqual(cfg.init_state.default_joint_angles, {'FL_hip_joint': .1})
        self.assertTrue(cfg.terrain.curriculum)
        self.assertEqual(cfg.terrain.num_sub_terrains, 200)
        self.assertEqual(saved, before)
        cfg.control.stiffness['joint'] = 9.
        self.assertEqual(saved['control']['stiffness']['joint'], 20.)

    def test_top_level_mapping_is_replaced_without_attribute_assignment(self):
        cfg = {'stale': 1}
        MODULE.overlay(cfg, {'joint': 20.})
        self.assertEqual(cfg, {'joint': 20.})
