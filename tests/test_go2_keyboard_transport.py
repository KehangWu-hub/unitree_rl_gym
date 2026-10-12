"""真实PTY按键经过父进程的本地消息路径；模拟子进程，不初始化DDS。"""
import contextlib
import io
import json
import os
from pathlib import Path
import pty
import socket
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from deploy.deploy_real import go2_commission as commission
from deploy.deploy_real import go2_keyboard as entry
from deploy.deploy_real.deploy_real_go2 import Go2RealConfig, TerminalKeys

ROOT = Path(__file__).resolve().parents[1]


class KeyboardTransportTest(unittest.TestCase):
    def exercise(self, end_key, failure=None):
        config = Go2RealConfig.load(entry.CONFIG, str(ROOT))
        master, slave = pty.openpty()
        messages, errors = [], []
        done = threading.Event()
        stdin = os.fdopen(os.dup(slave), 'r')
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/'control.sock')
            server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            server.bind(path)
            server.settimeout(3.)

            def receive():
                saw_zero = False
                try:
                    while not done.is_set():
                        raw, address = server.recvfrom(4096)
                        message = json.loads(raw); messages.append(message)
                        if message['type'] == 'status':
                            server.sendto(b'{"ready":true}', address)
                        elif message['type'] == 'arm':
                            os.write(master, b'W')
                        elif message['type'] == 'stop':
                            server.sendto(b'{"stopped":true}', address)
                            done.set()
                        elif message['type'] == 'heartbeat':
                            if 'w' in message.get('keys',[]):
                                os.write(master,end_key.encode())
                            elif any(key in message.get('keys',[]) for key in ('x','c')):
                                saw_zero = True
                            elif saw_zero:
                                # X后父进程仍发送心跳，不能退出或请求阻尼。
                                done.set()
                except BaseException as error:
                    errors.append(error);done.set()

            class Process:
                returncode = 0
                def poll(self): return 0 if done.is_set() else None
                def wait(self, timeout):
                    if not done.wait(timeout): raise subprocess.TimeoutExpired('mock',timeout)
                    return 0

            class Keys(TerminalKeys):
                def poll(self):
                    result=super().poll()
                    if failure is not None and 'x' in result:raise failure('simulated terminal failure')
                    return result

            thread = threading.Thread(target=receive,daemon=True)
            try:
                with patch.object(commission.sys,'stdin',stdin):
                    args = commission.parse_args(entry.commission_arguments('all','eno1',Path('unused.jsonl')))
                thread.start()
                # 首次接口存在检查是启动前的检查；模拟子进程已预先建立本地socket。
                with patch.object(commission,'default_socket',return_value=path), \
                     patch.object(commission.os.path,'lexists',return_value=False), \
                     patch.object(commission.subprocess,'Popen',return_value=Process()), \
                     patch.object(commission.sys,'stdin',stdin), patch.object(commission,'TerminalKeys',Keys), \
                     contextlib.redirect_stdout(io.StringIO()):
                    if failure is None:commission.run(args,config)
                    else:
                        with self.assertRaises(failure):commission.run(args,config)
                thread.join(3.)
                self.assertFalse(thread.is_alive())
                self.assertFalse(errors,repr(errors))
            finally:
                done.set();server.close();stdin.close();os.close(master);os.close(slave)
        return messages

    def test_x_is_forwarded_and_does_not_request_damping(self):
        messages=self.exercise('X')
        self.assertTrue(any('x' in m.get('keys',[]) for m in messages if m['type']=='heartbeat'))
        self.assertFalse(any(m['type']=='stop' for m in messages))

    def test_c_is_forwarded_with_heartbeat_and_does_not_request_damping(self):
        messages=self.exercise('C')
        self.assertTrue(any('c' in m.get('keys',[]) for m in messages if m['type']=='heartbeat'))
        self.assertFalse(any(m['type']=='stop' for m in messages))

    def test_z_stop_has_explicit_keyboard_source(self):
        messages=self.exercise('z')
        stop=next(m for m in messages if m['type']=='stop')
        self.assertEqual(stop['source'],'keyboard_z')
        self.assertEqual(stop['keys'],['z'])

    def test_ctrl_c_and_parent_error_are_distinguishable(self):
        for failure,source in [(KeyboardInterrupt,'ctrl_c'),(RuntimeError,'parent_error')]:
            with self.subTest(source=source):
                stop=next(m for m in self.exercise('x',failure) if m['type']=='stop')
                self.assertEqual(stop['source'],source)
                self.assertIn(failure.__name__,stop['error'])


if __name__ == '__main__':
    unittest.main()
