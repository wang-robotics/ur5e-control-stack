"""轨迹生成: PTP(关节空间点到点) 与 LINE(笛卡尔直线)。

所有轨迹段共享接口:
- sample(t) -> (q, qd, qdd)   t in [0, duration]
- duration: 段时长

速度规划:
- PTP: 五阶多项式 (quintic), 起停速度/加速度均为 0, 关节速度自动受限
- LINE: 位置用 quintic 平滑的直线, 姿态用 SLERP 同步插值,
        每个采样点过数值 IK (初值取上一采样解, 保证连续)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from . import config

# 逆解回调: (pos(3,), quat(4,) wxyz, q0(6,)) -> (q(6,), success)
IKFn = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, bool]]


def quintic_scale(t: float, T: float) -> tuple[float, float, float]:
    """五阶多项式平滑比例 s(t) 及其一/二阶导数。

    s(0)=0, s(T)=1, s'(0)=s'(T)=s''(0)=s''(T)=0。
    """
    if T <= 0:
        raise ValueError(f"时长必须为正, 收到 {T}")
    u = np.clip(t / T, 0.0, 1.0)
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / T
    dds = (60 * u - 180 * u**2 + 120 * u**3) / T**2
    return s, ds, dds


class TrajectorySegment(ABC):
    """轨迹段基类。"""

    def __init__(self, duration: float) -> None:
        if duration <= 0:
            raise ValueError(f"duration 必须为正, 收到 {duration}")
        self.duration = duration

    @abstractmethod
    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """采样 t 时刻的 (q, qd, qdd)。t 越界时裁剪到端点。"""


class PTPSegment(TrajectorySegment):
    """关节空间点到点 (Point-To-Point): 六轴同步 quintic 插补。"""

    def __init__(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        duration: Optional[float] = None,
        joint_speed: float = config.DEFAULT_JOINT_SPEED,
    ) -> None:
        self.q_start = np.asarray(q_start, dtype=float)
        self.q_goal = np.asarray(q_goal, dtype=float)
        if self.q_start.shape != self.q_goal.shape:
            raise ValueError("q_start 与 q_goal 维度不一致")
        self._delta = self.q_goal - self.q_start
        if duration is None:
            # 按最大关节位移 / 关节速度 自动定时长, 六轴同步到达
            max_q = float(np.max(np.abs(self._delta)))
            duration = max_q / max(joint_speed, 1e-6)
        super().__init__(duration)

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s, ds, dds = quintic_scale(t, self.duration)
        q = self.q_start + self._delta * s
        qd = self._delta * ds
        qdd = self._delta * dds
        return q, qd, qdd


class LineSegment(TrajectorySegment):
    """笛卡尔直线: TCP 沿直线运动, 姿态 SLERP 同步, 逐点 IK 成关节角。

    姿态语义: 给定 quat_start/quat_goal 做 SLERP; 只给 quat_start
    (quat_goal=None) 时保持起始姿态; 两者皆无时仅约束位置 (姿态自由)。
    """

    def __init__(
        self,
        p_start: np.ndarray,
        p_goal: np.ndarray,
        quat_start: Optional[np.ndarray] = None,
        quat_goal: Optional[np.ndarray] = None,
        ik_fn: Optional[IKFn] = None,
        q0: Optional[np.ndarray] = None,
        duration: Optional[float] = None,
        line_speed: float = config.DEFAULT_LINE_SPEED,
    ) -> None:
        self.p_start = np.asarray(p_start, dtype=float)
        self.p_goal = np.asarray(p_goal, dtype=float)
        self._delta_p = self.p_goal - self.p_start
        if duration is None:
            duration = float(np.linalg.norm(self._delta_p)) / max(
                line_speed, 1e-6
            )
        super().__init__(duration)

        # 姿态语义:
        #   quat_start + quat_goal  -> SLERP 插值
        #   quat_start, 无 quat_goal -> 保持起始姿态 (LINE 常见默认)
        #   无 quat_start, 有 quat_goal -> 非法 (无插值起点)
        if quat_goal is not None and quat_start is None:
            raise ValueError("提供 quat_goal 时必须同时提供 quat_start")
        if quat_start is not None and quat_goal is None:
            quat_goal = quat_start

        self.rot_start = (
            Rotation.from_quat(np.roll(quat_start, -1))
            if quat_start is not None
            else None
        )
        self.rot_goal = (
            Rotation.from_quat(np.roll(quat_goal, -1))
            if quat_goal is not None
            else None
        )
        self.ik_fn = ik_fn
        self._q_prev = q0

    def _pose_at(self, s: float) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """比例 s 处的 (pos, quat_wxyz 或 None)。"""
        p = self.p_start + self._delta_p * s
        if self.rot_start is not None:
            # SLERP
            slerp = Slerp(
                [0.0, 1.0],
                Rotation.concatenate([self.rot_start, self.rot_goal]),
            )
            rot = slerp([s])[0]
            quat = np.roll(rot.as_quat(), 1)  # xyzw -> wxyz
            return p, quat
        return p, None

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.ik_fn is None:
            raise RuntimeError("LineSegment 需要 ik_fn 才能输出关节角")
        s, _, _ = quintic_scale(t, self.duration)
        p, quat = self._pose_at(s)
        q, ok = self.ik_fn(p, quat, self._q_prev)
        if not ok:
            # IK 失败时保持上一解, 避免发散
            raise RuntimeError(f"LINE 轨迹 IK 失败 @ t={t:.4f}, pos={p}")
        self._q_prev = q
        return q, np.zeros_like(q), np.zeros_like(q)
