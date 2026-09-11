"""轨迹插补测试: PTP 边界条件、LINE 直线性。"""
import unittest

import numpy as np

from arm_ctrl.trajectory import LineSegment, PTPSegment, quintic_scale


class TestQuintic(unittest.TestCase):
    def test_boundary_conditions(self):
        T = 2.0
        s0, ds0, dds0 = quintic_scale(0.0, T)
        sT, dsT, ddsT = quintic_scale(T, T)
        self.assertAlmostEqual(s0, 0.0)
        self.assertAlmostEqual(sT, 1.0)
        self.assertAlmostEqual(ds0, 0.0)
        self.assertAlmostEqual(dsT, 0.0)
        self.assertAlmostEqual(dds0, 0.0)
        self.assertAlmostEqual(ddsT, 0.0)

    def test_monotonic(self):
        T = 2.0
        vals = [quintic_scale(t, T)[0] for t in np.linspace(0, T, 100)]
        self.assertTrue(all(a <= b for a, b in zip(vals, vals[1:])))


class TestPTPSegment(unittest.TestCase):
    def test_endpoints_and_velocity(self):
        q0 = np.array([0.0, -1.5, 0.0, -1.5, 0.0, 0.0])
        qf = np.array([1.0, -0.8, 1.2, -0.4, 0.7, -1.0])
        seg = PTPSegment(q0, qf, duration=2.0)

        q_a, qd_a, qdd_a = seg.sample(0.0)
        q_b, qd_b, qdd_b = seg.sample(2.0)
        np.testing.assert_allclose(q_a, q0, atol=1e-12)
        np.testing.assert_allclose(q_b, qf, atol=1e-12)
        np.testing.assert_allclose(qd_a, 0.0, atol=1e-12)
        np.testing.assert_allclose(qd_b, 0.0, atol=1e-12)
        np.testing.assert_allclose(qdd_a, 0.0, atol=1e-12)
        np.testing.assert_allclose(qdd_b, 0.0, atol=1e-12)

    def test_auto_duration(self):
        q0 = np.zeros(6)
        qf = np.array([1.0, 0, 0, 0, 0, 0])
        seg = PTPSegment(q0, qf, joint_speed=0.5)
        self.assertAlmostEqual(seg.duration, 2.0)

    def test_velocity_within_limit(self):
        """自动定时的 PTP 关节速度峰值不超过配置限速。"""
        q0 = np.zeros(6)
        qf = np.array([1.5, -1.0, 1.0, -0.8, 0.6, -0.5])
        seg = PTPSegment(q0, qf, joint_speed=0.8)
        ts = np.linspace(0, seg.duration, 200)
        max_v = max(np.max(np.abs(seg.sample(t)[1])) for t in ts)
        # quintic 峰值速度 = 1.875 * delta / T, 而 T 由 delta/0.8 决定
        self.assertLessEqual(max_v, 0.8 * 1.875 + 1e-6)


class TestLineSegment(unittest.TestCase):
    def test_pose_interpolation_straight(self):
        """无姿态约束时, 位置插值应严格共线。"""
        p0 = np.array([0.3, -0.1, 0.4])
        p1 = np.array([0.3, 0.2, 0.2])

        def ik_fn(p, quat, q_prev):
            return np.zeros(6), True

        seg = LineSegment(p0, p1, ik_fn=ik_fn, q0=np.zeros(6), duration=1.0)
        direction = p1 - p0
        direction /= np.linalg.norm(direction)
        for t in np.linspace(0, 1.0, 20):
            p, quat = seg._pose_at(quintic_scale(t, 1.0)[0])
            self.assertIsNone(quat)
            # 到直线距离应 ~0
            d = np.linalg.norm(np.cross(p - p0, direction))
            self.assertLess(d, 1e-9)

    def test_ik_called_with_prev_solution(self):
        """IK 初值应使用上一个采样点的解 (保证连续性)。"""
        calls = []

        def ik_fn(p, quat, q_prev):
            calls.append(None if q_prev is None else q_prev.copy())
            return np.zeros(6), True

        seg = LineSegment(
            np.zeros(3), np.ones(3), ik_fn=ik_fn, q0=None, duration=1.0
        )
        seg.sample(0.1)
        seg.sample(0.2)
        self.assertEqual(len(calls), 2)
        # 第一次初值为 None (构造时 q0=None), 第二次为上次的解
        self.assertIsNone(calls[0])
        np.testing.assert_array_equal(calls[1], np.zeros(6))


if __name__ == "__main__":
    unittest.main()
