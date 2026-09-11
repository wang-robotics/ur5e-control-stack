"""Gymnasium 环境与 RobotInterface 冒烟测试 (无渲染)。"""
import unittest

import numpy as np

from arm_ctrl import config
from arm_ctrl.envs import UR5eReachEnv
from arm_ctrl.interfaces import MuJoCoInterface


class TestMuJoCoInterface(unittest.TestCase):
    def test_reset_and_state(self):
        iface = MuJoCoInterface()
        iface.connect()
        q = iface.get_joint_positions()
        self.assertEqual(q.shape, (6,))
        np.testing.assert_allclose(q, config.HOME_QP0, atol=1e-6)

    def test_move_joints_blocking(self):
        iface = MuJoCoInterface()
        target = config.HOME_QP0 + np.array([0.5, -0.3, 0.4, 0.2, -0.3, 0.3])
        ok = iface.move_joints(target, duration=1.5, blocking=True)
        self.assertTrue(ok)
        self.assertFalse(iface.is_moving())
        # 位置伺服 (20Hz 带宽) 在 1.5s 内应基本到位 (允许小稳态误差)
        err = np.max(np.abs(iface.get_joint_positions() - target))
        self.assertLess(err, 0.05)

    def test_move_linear(self):
        iface = MuJoCoInterface()
        pos0, _ = iface.get_tcp_pose()
        goal = pos0 + np.array([0.05, 0.0, -0.05])
        ok = iface.move_linear(goal, duration=1.0, blocking=True)
        self.assertTrue(ok)
        pos1, _ = iface.get_tcp_pose()
        self.assertLess(np.linalg.norm(pos1 - goal), 0.01)

    def test_osc_velocity_streaming(self):
        """OSC 流控: 持续下发 +x 速度, TCP 应显著向 +x 移动。"""
        iface = MuJoCoInterface()
        pos0, _ = iface.get_tcp_pose()
        for _ in range(50):  # 1 s
            iface.set_tcp_velocity([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
            iface.step()
        pos1, _ = iface.get_tcp_pose()
        self.assertGreater(pos1[0] - pos0[0], 0.02)


class TestUR5eReachEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = UR5eReachEnv()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_reset_obs_shape(self):
        obs, info = self.env.reset(seed=0)
        self.assertEqual(obs.shape, (21,))
        self.assertEqual(self.env.observation_space.shape, (21,))
        self.assertEqual(self.env.action_space.shape, (6,))

    def test_random_rollout(self):
        self.env.reset(seed=1)
        total = 0.0
        for _ in range(20):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.assertEqual(obs.shape, (21,))
            self.assertFalse(terminated or truncated, "20 步内不应截断")
            total += reward
        self.assertTrue(np.isfinite(total))

    def test_success_terminates(self):
        """目标就在 TCP 处时应立即成功终止。"""
        self.env.reset(seed=2)
        tcp, _ = self.env.interface.get_tcp_pose()
        self.env.goal_pos = tcp.copy()  # 人为把目标放到当前位置
        _, reward, terminated, truncated, info = self.env.step(
            np.zeros(6)
        )
        self.assertTrue(terminated)
        self.assertTrue(info["success"])
        self.assertGreater(reward, 0.0)

    def test_horizon_truncation(self):
        """步数达到 horizon 后 truncated 为 True。"""
        self.env.reset(seed=3)
        truncated = False
        for _ in range(self.env.horizon + 1):
            _, _, _, truncated, _ = self.env.step(np.zeros(6))
        self.assertTrue(truncated)


if __name__ == "__main__":
    unittest.main()
