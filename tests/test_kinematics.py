"""运动学测试: FK/IK 一致性、雅可比方向正确性。"""
import unittest

import numpy as np

from arm_ctrl import config
from arm_ctrl.robot import Robot


class TestKinematics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = Robot()

    def test_fk_ik_consistency(self):
        """随机关节角 -> FK -> IK(同初值) -> FK' 应回到原位姿。"""
        rng = np.random.default_rng(0)
        for _ in range(10):
            q = rng.uniform(-2.0, 2.0, size=6)
            pos, quat = self.robot.fk(q)
            q_ik, ok = self.robot.ik(pos, quat, q0=q)
            self.assertTrue(ok, f"IK 失败 @ q={q}")
            pos2, quat2 = self.robot.fk(q_ik)
            self.assertLess(
                np.linalg.norm(pos2 - pos),
                2e-3,
                f"位置不一致: {pos} vs {pos2}",
            )
            # 姿态误差用旋转角衡量 (< 0.5°)
            from scipy.spatial.transform import Rotation

            R1 = Rotation.from_quat(np.roll(quat, -1))
            R2 = Rotation.from_quat(np.roll(quat2, -1))
            ang_err = Rotation.from_matrix(R1.as_matrix() @ R2.as_matrix().T)
            self.assertLess(
                float(np.linalg.norm(ang_err.as_rotvec())),
                np.deg2rad(0.5),
            )

    def test_ik_position_only_from_home(self):
        """只约束位置时, 从 home 出发 IK 到前方工作点。"""
        pos, quat_home = self.robot.fk(config.HOME_QP0)
        target = pos + np.array([0.1, 0.05, -0.15])
        q_ik, ok = self.robot.ik(target, None, q0=config.HOME_QP0)
        self.assertTrue(ok, "位置 IK 失败")
        pos2, _ = self.robot.fk(q_ik)
        self.assertLess(np.linalg.norm(pos2 - target), 2e-3)

    def test_jacobian_motion_direction(self):
        """OSC 关节速度积分后, TCP 应沿目标线速度方向移动。"""
        q = config.HOME_QP0.copy()
        v = np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])  # 纯 +x 线速度
        dq = self.robot.osc_joint_velocities(q, v)
        pos0, _ = self.robot.fk(q)
        pos1, _ = self.robot.fk(q + dq * 0.1)
        delta = pos1 - pos0
        # 位移方向与 v 前三维夹角应很小
        cos_ang = float(
            np.dot(delta, v[:3]) / (np.linalg.norm(delta) * np.linalg.norm(v[:3]))
        )
        self.assertGreater(cos_ang, 0.95, f"位移方向偏差过大: {delta}")

    def test_tcp_pose_consistency(self):
        """tcp_pose() 与 fk() 输出一致。"""
        self.robot.reset(config.HOME_QP0)
        pos, quat = self.robot.tcp_pose()
        pos_fk, quat_fk = self.robot.fk(config.HOME_QP0)
        np.testing.assert_allclose(pos, pos_fk, atol=1e-6)
        np.testing.assert_allclose(quat, quat_fk, atol=1e-6)

    def test_home_pose_reasonable(self):
        """home 位姿下 TCP 应在基座上方约 0.4~0.6 m 处。"""
        pos, _ = self.robot.fk(config.HOME_QP0)
        self.assertGreater(pos[2], 0.4)
        self.assertLess(pos[2], 0.6)


if __name__ == "__main__":
    unittest.main()
