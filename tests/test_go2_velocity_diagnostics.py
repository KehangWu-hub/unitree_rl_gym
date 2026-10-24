"""The offline inspection path must not initialize DDS or a motor transport."""
import contextlib
import io
import sys
import types
import unittest
from unittest.mock import patch

from deploy.deploy_real import go2_velocity_diagnostics as diagnostics


class VelocityDiagnosticsTest(unittest.TestCase):
    def test_check_inspects_fields_without_constructing_transport(self):
        module = types.ModuleType("unitree_sdk2py.idl.unitree_go.msg.dds_")
        module.SportModeState_ = type("SportModeState_", (), {"__annotations__": {"velocity": list}})
        with patch.dict(sys.modules, {module.__name__: module}), \
             patch.object(sys, "argv", ["diagnostics", "--check"]), \
             patch.object(diagnostics, "VerifiedTransport", side_effect=AssertionError("must remain offline")), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(diagnostics.main(), 0)
        self.assertIn('"dds_initialized": false', output.getvalue())
        self.assertIn('"velocity"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
