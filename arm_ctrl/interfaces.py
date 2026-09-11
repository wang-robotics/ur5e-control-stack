"""通信抽象层: RobotInterface 抽象基类 + MuJoCo 仿真实现。

设计目标: 上层 (轨迹执行器、RL 环境) 只依赖 RobotInterface,
不接触 MuJoCo API。将来接入真机 (ROS2 / gRPC / 串口) 时,
只需新增一个实现类, 控制栈与 RL 环境零改动。

RobotInterface 提供两级控制入口:
- 轨迹级: move_joints / move_linear   (给定目标, 内部完成规划与执行)
- 流控级: set_joint_target / set_tcp_velocity  (每周期下发命令,
  供 RL 或外部遥操作使用)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import mujoco
import numpy as np

from . import config
from .controller import (
    JointPositionController,
    OSCController,
    set_position_servo,
)
from .robot import Robot
from .trajectory import LineSegment, PTPSegment, TrajectorySegment


class RobotInterface(ABC):
    """机械臂控制抽象接口 (仿真与真机通用)。"""

    # ------------------------------------------------------------ 生命周期
    @abstractmethod
    def connect(self) -> bool:
        """建立通信连接, 返回是否成功。"""

    @abstractmethod
    def disconnect(self) -> None:
        """关闭连接并安全停机。"""

    # ------------------------------------------------------------ 状态读取
    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """关节角 (rad), 形状 (6,)。"""

    @abstractmethod
    def get_joint_velocities(self) -> np.ndarray:
        """关节角速度 (rad/s), 形状 (6,)。"""

    @abstractmethod
    def get_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """末端位姿: (pos(3,), quat(4,) wxyz)。"""

    # ------------------------------------------------------------ 轨迹级命令
    @abstractmethod
    def move_joints(
        self,
        q_goal: np.ndarray,
        duration: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """关节空间点到点 (PTP)。blocking=False 时命令挂起, 由 step() 推进。"""

    @abstractmethod
    def move_linear(
        self,
        pos_goal: np.ndarray,
        quat_goal: Optional[np.ndarray] = None,
        duration: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """笛卡尔直线 (LINE)。"""

    # ------------------------------------------------------------ 流控级命令
    @abstractmethod
    def set_joint_target(self, q: np.ndarray) -> None:
        """设置关节位置目标 (位置流控, 每周期覆盖)。"""

    @abstractmethod
    def set_tcp_velocity(self, twist: np.ndarray) -> None:
        """设置末端 6 维速度 (OSC 流控, RL task-space 动作入口)。"""

    @abstractmethod
    def stop(self) -> None:
        """取消当前命令, 停在当前位置。"""

    @abstractmethod
    def is_moving(self) -> bool:
        """是否有轨迹正在执行 (未完成)。"""

    # ------------------------------------------------------------ 执行推进
    @abstractmethod
    def step(self) -> None:
        """推进一个控制周期 (CTRL_DT)。真机实现中为空操作或通信轮询。"""

    @abstractmethod
    def reset(self, q: Optional[np.ndarray] = None) -> None:
        """复位到 home (或指定位姿), 清空所有挂起命令。"""


class MuJoCoInterface(RobotInterface):
    """RobotInterface 的 MuJoCo 仿真实现。"""

    def __init__(
        self,
        robot: Optional[Robot] = None,
        servo_kv_ratio: Optional[float] = 0.05,
    ) -> None:
        """构造 MuJoCo 接口实现。

        Args:
            robot: Robot 实例; None 时自动加载 UR5e 模型。
            servo_kv_ratio: 位置伺服阻尼比 kv/kp。None 保持模型原值
                (带宽 0.8Hz, 跟踪偏慢); 0.05 为推荐值 (~20Hz 带宽)。
        """
        self.robot = robot if robot is not None else Robot()
        if servo_kv_ratio is not None:
            set_position_servo(self.robot.model, kv_ratio=servo_kv_ratio)
        self._pos_ctrl = JointPositionController(self.robot)
        self._osc_ctrl = OSCController(self.robot)
        self._segment: Optional[TrajectorySegment] = None
        self._seg_t = 0.0
        self._connected = False
        self.reset()

    # ------------------------------------------------------------ 生命周期
    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self.stop()
        self._connected = False

    # ------------------------------------------------------------ 状态读取
    def get_joint_positions(self) -> np.ndarray:
        return self.robot.qpos

    def get_joint_velocities(self) -> np.ndarray:
        return self.robot.qvel

    def get_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.robot.tcp_pose()

    # ------------------------------------------------------------ 轨迹级命令
    def move_joints(
        self,
        q_goal: np.ndarray,
        duration: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        self._osc_ctrl.clear_target()
        seg = PTPSegment(self.robot.qpos, q_goal, duration)
        if blocking:
            return self._run_segment_blocking(seg)
        self._segment = seg
        self._seg_t = 0.0
        return True

    def move_linear(
        self,
        pos_goal: np.ndarray,
        quat_goal: Optional[np.ndarray] = None,
        duration: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        self._osc_ctrl.clear_target()
        pos, quat = self.robot.tcp_pose()
        q0 = self.robot.qpos

        def ik_fn(p, qu, q_prev):
            return self.robot.ik(p, qu, q_prev)

        seg = LineSegment(
            pos,
            pos_goal,
            quat_start=quat,
            quat_goal=quat_goal,
            ik_fn=ik_fn,
            q0=q0,
            duration=duration,
        )
        if blocking:
            return self._run_segment_blocking(seg)
        self._segment = seg
        self._seg_t = 0.0
        return True

    def _run_segment_blocking(self, seg: TrajectorySegment) -> bool:
        self._segment = seg
        self._seg_t = 0.0
        while self._seg_t < seg.duration:
            self.step()
        return True

    # ------------------------------------------------------------ 流控级命令
    def set_joint_target(self, q: np.ndarray) -> None:
        self._segment = None
        self._osc_ctrl.clear_target()
        self._pos_ctrl.set_target(q)

    def set_tcp_velocity(self, twist: np.ndarray) -> None:
        self._segment = None
        # 从"未激活"重新激活 OSC 时, 把积分目标同步到当前位置, 防止跳变
        if not self._osc_active():
            self._osc_ctrl.q_des = self.robot.qpos.copy()
        self._osc_ctrl.set_target_twist(twist)

    def stop(self) -> None:
        self._segment = None
        self._osc_ctrl.clear_target()
        # 保持当前位置: 以当前角为目标
        self._pos_ctrl.set_target(self.robot.qpos)

    def is_moving(self) -> bool:
        return self._segment is not None

    # ------------------------------------------------------------ 执行推进
    def step(self) -> None:
        if not self._connected:
            self.connect()

        if self._segment is not None:
            self._seg_t += config.CTRL_DT
            if self._seg_t >= self._segment.duration:
                q, _, _ = self._segment.sample(self._segment.duration)
                self._segment = None
            else:
                q, _, _ = self._segment.sample(self._seg_t)
            self._pos_ctrl.set_target(q)

        # 位置轨迹与 OSC 统一走位置命令; OSC 激活时以它为准
        ctrl = (
            self._osc_ctrl.compute()
            if self._segment is None and self._osc_active()
            else self._pos_ctrl.compute()
        )
        self.robot.data.ctrl[:] = ctrl

        # 物理子步
        mujoco.mj_step(self.robot.model, self.robot.data, config.STEPS_PER_CTRL)

    def _osc_active(self) -> bool:
        """OSC 是否处于激活状态 (有非零速度命令)。"""
        return bool(np.any(np.abs(self._osc_ctrl._twist) > 0.0))

    def reset(self, q: Optional[np.ndarray] = None) -> None:
        self.robot.reset(q)
        self._segment = None
        self._seg_t = 0.0
        self._pos_ctrl.reset()
        self._pos_ctrl.set_target(self.robot.qpos)
        self._osc_ctrl.reset()

    # ------------------------------------------------------------ 仿真特有
    @property
    def model(self) -> mujoco.MjModel:
        """渲染/调试用, 真机实现中不存在该属性。"""
        return self.robot.model

    @property
    def data(self) -> mujoco.MjData:
        return self.robot.data
