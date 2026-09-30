import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import yaml

from deploy.deploy_real.deploy_real_go2 import Go2Mode, Go2MotorTest, Go2RealConfig, Go2State, parse_args
from deploy.deploy_real.go2_watchdog import CommandGate, default_socket, request, require_simulation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "deploy/deploy_real/configs/go2.yaml"


class Go2WatchdogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Go2RealConfig.load(CONFIG, str(ROOT))

    def state(self, now):
        return Go2State(now, np.array([1, 0, 0, 0]), np.zeros(3),
                        self.cfg.default_joint_pos[self.cfg.sdk_to_policy].copy(), np.zeros(12))

    def payload(self, now, mode=Go2Mode.FIX_STAND):
        return {"time": now, "mode": mode.value, "target": self.state(now).joint_pos_sdk.tolist(),
                "kp": [5.0] * 12, "kd": [0.5] * 12}

    def test_lease_expiry_latches_and_ignores_later_commands(self):
        gate = CommandGate(self.cfg, 2.0)
        gate.accept(self.payload(1.0), 1.0, self.state(1.0))
        self.assertEqual(gate.output(1.05, self.state(1.05)).mode, Go2Mode.FIX_STAND)
        stopped = gate.output(1.11, self.state(1.11))
        self.assertEqual(stopped.mode, Go2Mode.DAMPING)
        np.testing.assert_array_equal(stopped.kp_sdk, 0)
        self.assertIn("心跳超时", stopped.reason)
        gate.accept(self.payload(1.12), 1.12, self.state(1.12))
        self.assertEqual(gate.output(1.12, self.state(1.12)).reason, stopped.reason)

    def test_invalid_commands_state_and_hard_duration_stop(self):
        for kind in ("gain", "target", "nan", "timestamp", "stale_state", "duration"):
            with self.subTest(kind=kind):
                gate = CommandGate(self.cfg, 0.03)
                payload = self.payload(1.0)
                if kind == "gain":
                    payload["kp"][0] = 100
                elif kind == "target":
                    payload["target"][0] = 9
                elif kind == "nan":
                    payload["target"][0] = float("nan")
                elif kind == "timestamp":
                    payload["time"] = float("nan")
                gate.accept(payload, 1.0, self.state(1.0))
                now = 1.04 if kind == "duration" else 1.01
                state = self.state(0.5 if kind == "stale_state" else now)
                output = gate.output(now, state)
                self.assertEqual(output.mode, Go2Mode.DAMPING)
                np.testing.assert_array_equal(output.kp_sdk, 0)

    def test_motor_test_single_joint_bounds_and_no_policy(self):
        test = Go2MotorTest(self.cfg, "FL_hip_joint", 0.01, 2.0)
        first = test.step(1.0, self.state(1.0))
        peak = None
        for tick in range(1, 100):
            now = 1.0 + tick * 0.02
            cmd = test.step(now, self.state(now))
            self.assertEqual(cmd.mode, Go2Mode.MOTOR_TEST)
            delta = cmd.target_pos_sdk - first.target_pos_sdk
            self.assertLessEqual(np.max(np.abs(delta)), 0.010001)
            self.assertEqual(np.flatnonzero(np.abs(delta) > 1e-6).tolist(), [3])
            if tick == 50:
                peak = delta[3]
        self.assertAlmostEqual(peak, 0.01, places=5)
        self.assertEqual(test.step(3.0, self.state(3.0)).mode, Go2Mode.DAMPING)
        gate = CommandGate(self.cfg, 2.0, motor_test=True)
        payload = self.payload(1.0, Go2Mode.MOTOR_TEST)
        payload["target"][0] += 0.03
        gate.accept(payload, 1.0, self.state(1.0))
        self.assertEqual(gate.output(1.01, self.state(1.01)).mode, Go2Mode.DAMPING)

    def test_real_interfaces_and_invalid_parameters_rejected(self):
        for network, domain, duration in (("eno1", 1, 2), ("lo", 0, 2), ("lo", 233, 2),
                                          ("lo", 1, float("nan")), ("lo", 1, 61)):
            with self.subTest(network=network, domain=domain, duration=duration), self.assertRaises(ValueError):
                require_simulation(network, domain, duration)
        for amplitude, duration in ((0.021, 2), (0.01, 4), (float("nan"), 2), (0, 2)):
            with self.assertRaises(ValueError):
                Go2MotorTest(self.cfg, "FL_hip_joint", amplitude, duration)
        args = parse_args(["lo", "--domain-id", "1", "--motor-test"])
        self.assertEqual(args.duration, 2.0)


@unittest.skipUnless(os.environ.get("GO2_DDS_TESTS") == "1", "设置GO2_DDS_TESTS=1执行动态DDS故障注入")
class Go2DDSWatchdogIntegrationTest(unittest.TestCase):
    def rows(self, directory):
        path = directory / "trace.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.endswith("}")]

    def wait_for(self, predicate, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                return result
            time.sleep(0.01)
        self.fail("等待DDS测试条件超时")

    def test_dynamic_mujoco_motor_and_independent_stop(self):
        summaries = []
        for index, scenario in enumerate(("motor", "ground_motor", "terminal_stop", "sigstop", "sigkill", "state_loss")):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory(prefix="go2-dds-test-") as tmp:
                directory = Path(tmp)
                domain = 100 + os.getpid() % 90 + index
                fixture_log = open(directory / "fixture.log", "w")
                runner_log = open(directory / "runner.log", "w")
                fixture = runner = None
                try:
                    fixture_cmd = [sys.executable, "-m", "tests.go2_dds_fixture", "--domain-id", str(domain),
                                   "--directory", tmp]
                    if scenario == "motor":
                        fixture_cmd.append("--supported")
                    fixture = subprocess.Popen(fixture_cmd, cwd=ROOT, stdout=fixture_log, stderr=subprocess.STDOUT)
                    self.wait_for(lambda: (directory / "ready").exists())
                    cfg = copy.deepcopy(yaml.safe_load(CONFIG.read_text()))
                    cfg["recording"]["directory"] = str(directory / "records")
                    cfg_path = directory / "config.yaml"
                    cfg_path.write_text(yaml.safe_dump(cfg))
                    runner_cmd = [sys.executable, "-m", "deploy.deploy_real.deploy_real_go2", "lo",
                                  "--domain-id", str(domain), "--config", str(cfg_path), "--duration",
                                  "2" if "motor" in scenario else "8"]
                    runner_cmd += ["--motor-test"] if "motor" in scenario else ["--simulation-auto", "--command", "0", "0", "0"]
                    runner = subprocess.Popen(runner_cmd, cwd=ROOT, stdout=runner_log, stderr=subprocess.STDOUT)
                    self.wait_for(lambda: any(r["type"] == "command" and max(r["kp"]) > 0 for r in self.rows(directory)))
                    watchdog_pid = request(default_socket(domain), {"type": "status"})["pid"]
                    if "motor" not in scenario:
                        # 等待实际策略帧出现（Policy增益20）；不是仅看进程启动文字。
                        self.wait_for(lambda: any(r["type"] == "command" and abs(max(r["kp"]) - 20) < 1e-6 for r in self.rows(directory)))
                    trigger = None
                    if scenario == "terminal_stop":
                        trigger = time.monotonic()
                        result = subprocess.run([sys.executable, "-m", "deploy.deploy_real.go2_watchdog", "--stop",
                                                 "--domain-id", str(domain)], cwd=ROOT, capture_output=True, text=True, timeout=3)
                        self.assertEqual(result.returncode, 0, result.stderr)
                    elif scenario in ("sigstop", "sigkill"):
                        trigger = time.monotonic()
                        os.kill(runner.pid, signal.SIGSTOP if scenario == "sigstop" else signal.SIGKILL)
                    elif scenario == "state_loss":
                        trigger = time.monotonic()
                        (directory / "pause_state").touch()
                    damping = self.wait_for(lambda: next((r for r in self.rows(directory)
                                                          if r["type"] == "command" and max(r["kp"]) == 0
                                                          and min(r["kd"]) == 3.0), None), timeout=6)
                    if trigger is not None:
                        self.assertLess(damping["time"] - trigger, 0.25)
                    # 完整观察锁存与一秒阻尼，冻结的主进程不参与这一步。
                    self.wait_for(lambda: not os.path.exists(default_socket(domain)), timeout=4)
                    if scenario == "sigstop":
                        os.kill(runner.pid, signal.SIGCONT)
                    code = runner.wait(timeout=5)
                    if scenario != "sigkill":
                        self.assertEqual(code, 0, (directory / "runner.log").read_text())
                    rows = self.rows(directory)
                    commands = [r for r in rows if r["type"] == "command"]
                    self.assertTrue(all(r["crc_valid"] and r["mode"] == [1] * 12 for r in commands))
                    self.assertTrue(all(max(r["kp"]) == 0 for r in commands if r["time"] >= damping["time"]))
                    self.assertGreater(commands[-1]["time"] - damping["time"], 0.90)
                    peak_target = None
                    if scenario == "motor":
                        active = [r for r in commands if max(r["kp"]) > 0]
                        initial = np.array(active[0]["target"])
                        offsets = np.array([r["target"] for r in active]) - initial
                        self.assertLessEqual(np.max(np.abs(offsets)), 0.010001)
                        self.assertGreater(np.max(offsets[:, 3]), 0.009)
                        self.assertLess(np.max(np.abs(offsets[:, [i for i in range(12) if i != 3]])), 1e-6)
                        states = [r for r in rows if r["type"] == "state" and active[0]["time"] <= r["time"] <= damping["time"]]
                        actual = np.array([r["q"] for r in states]) - initial
                        self.assertGreater(np.max(actual[:, 3]), 0.001)
                        self.assertLess(np.max(np.abs(actual)), 0.04)
                        peak_target = float(np.max(offsets[:, 3]))
                    summaries.append({"scenario": scenario, "watchdog_pid": watchdog_pid,
                                      "stop_latency_ms": None if trigger is None else round(1000 * (damping["time"] - trigger), 2),
                                      "damping_seconds": round(commands[-1]["time"] - damping["time"], 3),
                                      "peak_target_rad": peak_target,
                                      "runner_result": code, "runner_output": (directory / "runner.log").read_text()})
                finally:
                    if runner and runner.poll() is None:
                        os.kill(runner.pid, signal.SIGCONT)
                        runner.terminate()
                        try:
                            runner.wait(timeout=4)
                        except subprocess.TimeoutExpired:
                            runner.kill()
                            runner.wait()
                    if os.path.exists(default_socket(domain)):
                        try:
                            request(default_socket(domain), {"type": "stop"})
                            self.wait_for(lambda: not os.path.exists(default_socket(domain)), timeout=4)
                        except OSError:
                            pass
                    if fixture:
                        fixture.terminate()
                        fixture.wait(timeout=5)
                    fixture_log.close()
                    runner_log.close()
        report_path = os.environ.get("GO2_DDS_REPORT", "/tmp/go2_watchdog_report.json")
        Path(report_path).write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
        print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    unittest.main()
