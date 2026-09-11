"""UR5e 全局参数配置。

所有模块共享的常量集中在这里，避免魔法数字散落各处。
单位约定: 位置 m / 姿态四元数 wxyz / 角速度 rad/s / 时间 s。
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
# robot_descriptions 中 UR5e 的描述文件 (官方 Menagerie MJCF)
ROBOT_MODEL_NAME = "ur5e"
# 末端 TCP 参考点 (Menagerie UR5e 模型中的 site 名, 位于法兰盘末端)
TASK_SITE = "attachment_site"

# 关节名 (Menagerie UR5e 从基座到腕部, 新版本带 _joint 后缀)
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# UR5e 关节限位: 前 3 个大关节 ±360°, 后 3 个腕关节 ±180°
JOINT_LIMITS = np.array(
    [
        [-2.0 * np.pi, 2.0 * np.pi],
        [-2.0 * np.pi, 2.0 * np.pi],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
    ]
)

# home 位姿 (Menagerie UR5e keyframe "home")
HOME_QP0 = np.array(
    [-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0]
)

# 关节速度限位 (rad/s), 用于轨迹规划和 RL 安全裁剪
JOINT_VEL_LIMIT = 3.14
# 末端线速度 / 角速度限位 (m/s, rad/s), 用于 OSC 与 RL 动作裁剪
TCP_VEL_LIMIT = 0.15
TCP_ANGVEL_LIMIT = 0.8

# ---------------------------------------------------------------------------
# 仿真与控制
# ---------------------------------------------------------------------------
# MuJoCo 物理步长 (与 Menagerie 模型一致)
SIM_DT = 0.002
# 控制周期 (轨迹/OSC/RL 决策频率, 50 Hz)
CTRL_DT = 0.02
# 每个控制周期内的物理子步数
STEPS_PER_CTRL = int(round(CTRL_DT / SIM_DT))

# 轨迹默认速度 (rad/s 关节空间, m/s 笛卡尔空间)
DEFAULT_JOINT_SPEED = 0.8
DEFAULT_LINE_SPEED = 0.05

# ---------------------------------------------------------------------------
# 控制器
# ---------------------------------------------------------------------------
# 位置模式: 直接使用模型内建 position actuator
# PD 力矩模式增益 (力矩控制演示用)
PD_KP = np.array([300.0] * 6)
PD_KD = np.array([25.0] * 6)
# OSC 差分 IK 的阻尼最小二乘正则 (越大越稳、跟踪精度越低)
OSC_DLS_LAMBDA = 1e-2

# ---------------------------------------------------------------------------
# RL 环境
# ---------------------------------------------------------------------------
REACH_ACTION_SCALE = np.array(
    [TCP_VEL_LIMIT] * 3 + [TCP_ANGVEL_LIMIT] * 3
)
REACH_EPISODE_STEPS = 100          # 每个 episode 最大步数 (100 * 0.02s = 2s)
REACH_SUCCESS_DIST = 0.02          # 末端距目标小于该值视为成功 (m)
# 目标点相对初始 TCP 的采样偏移半径范围 (m) —— 保证 2s 内可达
REACH_GOAL_OFFSET = (0.08, 0.25)
REACH_GOAL_Z_MIN = 0.10            # 目标点最低高度 (桌面以上)
