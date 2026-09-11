"""关节控制器: 位置控制 / PD 力矩控制 / OSC 末端速度控制。

- JointPositionController: 利用模型内建 position actuator, ctrl 直接设目标角
- JointTorquePDController: 力矩模式 PD (需先调用 enable_torque_mode 把
  position actuator 的伺服增益清零, 使 ctrl 变为纯力矩)
- OSCController: 末端 6 维速度 -> 差分 IK -> 关节速度积分 -> 位置命令,
  即 RL 环境的 task-space 动作入口
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import mujoco
import numpy as np

from . import config
from .robot import Robot


def enable_torque_mode(model: mujoco.MjModel) -> None:
    """把模型内建 position actuator 切换为力矩模式。

    原理: Menagerie UR5e 的 general actuator 为位置伺服
    (gainprm=kp, biasprm=[b_ctrl, b_act, b_actdot])。
    将 kp 清零并把 bias 改为纯 ctrl 直通后, 输出力 = ctrl, 即纯力矩。

    恢复方法见 enable_position_mode()。
    """
    model.actuator_gainprm[:, 0] = 0.0
    model.actuator_biasprm[:, 0] = 1.0  # ctrl 直通
    model.actuator_biasprm[:, 1] = 0.0  # 去掉 act 项
    model.actuator_biasprm[:, 2] = 0.0  # 去掉 actdot 项


def set_position_servo(model: mujoco.MjModel, kv_ratio: float = 0.05) -> None:
    """调整位置伺服的阻尼比 kv/kp, 改变伺服带宽。

    Menagerie UR5e 内建伺服 kv = 0.2*kp, 闭环为一阶低通,
    时间常数 tau = kv/kp = 0.2s (带宽约 0.8Hz), 跟踪轨迹偏慢。
    减小 kv_ratio 可提高带宽 (0.05 -> tau=50ms, ~20Hz), 仍稳定无超调。
    """
    model.actuator_biasprm[:, 2] = -kv_ratio * model.actuator_gainprm[:, 0]


def enable_position_mode(model: mujoco.MjModel) -> None:
    """恢复模型内建 position actuator 的位置伺服增益。

    Menagerie UR5e: size3 关节 kp=2000, size1 腕关节 kp=500;
    biasprm = [0, -kp, -0.2*kp] 近似 (原模型为 [0, -2000, -400] 与
    [0, -500, -100])。恢复为伺服模式。
    """
    n = model.nu
    if n == 0:
        return
    # 按关节尺寸分档: 前三关节 2000, 后三 500 (与 Menagerie UR5e 定义一致)
    kp = np.ones(n) * 500.0
    kp[:3] = 2000.0
    model.actuator_gainprm[:, 0] = kp
    model.actuator_biasprm[:, 0] = 0.0
    model.actuator_biasprm[:, 1] = -kp
    model.actuator_biasprm[:, 2] = -0.2 * kp


class BaseController(ABC):
    """控制器基类: 每个控制周期输出一次 ctrl。"""

    def __init__(self, robot: Robot) -> None:
        self.robot = robot

    def reset(self) -> None:
        """清除控制器内部状态 (积分量等)。"""

    @abstractmethod
    def compute(self) -> np.ndarray:
        """基于 robot.data 当前状态计算 ctrl (形状 (nu,))。"""


class JointPositionController(BaseController):
    """位置控制: 直接下发关节目标角 (模型 position actuator)。"""

    def __init__(self, robot: Robot, q_target: Optional[np.ndarray] = None):
        super().__init__(robot)
        self.q_target = (
            robot.home_qpos.copy()
            if q_target is None
            else np.asarray(q_target, dtype=float)
        )

    def set_target(self, q: np.ndarray) -> None:
        self.q_target = np.clip(
            np.asarray(q, dtype=float),
            config.JOINT_LIMITS[:, 0],
            config.JOINT_LIMITS[:, 1],
        )

    def compute(self) -> np.ndarray:
        return self.q_target.copy()


class JointTorquePDController(BaseController):
    """力矩模式 PD + 重力补偿。

    使用前须调用 enable_torque_mode(robot.model)。
    tau = Kp(q_des - q) + Kd(qd_des - qd) + qfrc_bias
    """

    def __init__(
        self,
        robot: Robot,
        kp: np.ndarray = config.PD_KP,
        kd: np.ndarray = config.PD_KD,
    ) -> None:
        super().__init__(robot)
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.q_target = robot.home_qpos.copy()
        self.qd_target = np.zeros(robot.nv)

    def set_target(
        self, q: np.ndarray, qd: Optional[np.ndarray] = None
    ) -> None:
        self.q_target = np.clip(
            np.asarray(q, dtype=float),
            config.JOINT_LIMITS[:, 0],
            config.JOINT_LIMITS[:, 1],
        )
        if qd is not None:
            self.qd_target = np.asarray(qd, dtype=float)

    def compute(self) -> np.ndarray:
        data = self.robot.data
        tau = self.kp * (self.q_target - data.qpos[: self.robot.nq])
        tau += self.kd * (self.qd_target - data.qvel[: self.robot.nv])
        tau += data.qfrc_bias[: self.robot.nv]  # 重力补偿
        return tau


class OSCController(BaseController):
    """OSC: 末端速度命令 -> 关节位置命令 (RL task-space 动作入口)。

    流程 (每个控制周期):
        1. 读取当前关节角, 计算数值雅可比
        2. dq = DLS(J, lambda) @ twist
        3. q_des += dq * dt, 裁剪到关节限位
        4. 输出 q_des 给 position actuator 伺服
    """

    def __init__(self, robot: Robot) -> None:
        super().__init__(robot)
        self.q_des = robot.home_qpos.copy()
        self._twist = np.zeros(6)

    def reset(self) -> None:
        self.q_des = self.robot.home_qpos.copy()
        self._twist = np.zeros(6)

    def set_target_twist(self, twist: np.ndarray) -> None:
        """设置末端速度目标 [vx, vy, vz, wx, wy, wz]。"""
        twist = np.asarray(twist, dtype=float)
        twist[:3] = np.clip(twist[:3], -config.TCP_VEL_LIMIT, config.TCP_VEL_LIMIT)
        twist[3:] = np.clip(
            twist[3:], -config.TCP_ANGVEL_LIMIT, config.TCP_ANGVEL_LIMIT
        )
        self._twist = twist

    def clear_target(self) -> None:
        """清零速度命令, 使控制器不再输出新的关节目标。"""
        self._twist[:] = 0.0

    def compute(self) -> np.ndarray:
        q = self.robot.data.qpos[: self.robot.nv].copy()
        dq = self.robot.osc_joint_velocities(q, self._twist)
        self.q_des += dq * config.CTRL_DT
        self.q_des = np.clip(
            self.q_des,
            config.JOINT_LIMITS[:, 0],
            config.JOINT_LIMITS[:, 1],
        )
        return self.q_des.copy()
