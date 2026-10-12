"""恢复原厂前等待稳定状态；不创建DDS或发送命令。"""
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from deploy.deploy_real.go2_commission import check_resting_for_restore, wait_resting_for_restore
from deploy.deploy_real.deploy_real_go2 import Go2RealConfig, Go2State
ROOT=Path(__file__).resolve().parents[1]

class RestoreWaitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=Go2RealConfig.load(ROOT/'deploy/deploy_real/configs/go2.yaml',str(ROOT))

    def state(self,now,speed=0.,calf=-2.65):
        q=np.tile([0.,1.2,calf],4)
        return Go2State(now,np.array([1.,0,0,0]),np.zeros(3),q,np.full(12,speed))

    def test_transient_rejection_waits_then_passes_without_changing_limits(self):
        now=[0.]
        reader=SimpleNamespace(snapshot=lambda:self.state(now[0],.3 if now[0]<.1 else 0.))
        def sleep(dt):now[0]+=dt
        result=wait_resting_for_restore(self.config,reader,clock=lambda:now[0],sleep=sleep)
        self.assertGreaterEqual(now[0],1.1)
        self.assertLess(now[0],1.15)
        np.testing.assert_array_equal(result.joint_vel_sdk,0.)

    def test_invalid_pose_times_out_with_joint_detail(self):
        now=[0.]
        reader=SimpleNamespace(snapshot=lambda:self.state(now[0],calf=-1.3))
        def sleep(dt):now[0]+=dt
        with self.assertRaisesRegex(ValueError,'calf_joint.*-1.300'):
            wait_resting_for_restore(self.config,reader,timeout=.2,clock=lambda:now[0],sleep=sleep)
        with self.assertRaisesRegex(ValueError,'速度峰值=0.300'):
            check_resting_for_restore(self.config,self.state(0.,.3),0.)

    def test_stale_state_never_counts_as_stable(self):
        now=[0.]
        reader=SimpleNamespace(snapshot=lambda:self.state(0.))
        def sleep(dt):now[0]+=dt
        with self.assertRaisesRegex(ValueError,'LowState通信超时'):
            wait_resting_for_restore(self.config,reader,timeout=.3,clock=lambda:now[0],sleep=sleep)

if __name__=='__main__':unittest.main()
