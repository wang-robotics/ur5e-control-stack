"""arm_ctrl: MuJoCo UR5e 六轴机械臂控制栈。

模块组成:
- config:       全局参数配置
- robot:        模型加载 / 正运动学 FK / 逆运动学 IK / 雅可比
- trajectory:   轨迹生成 (PTP / LINE)
- controller:   关节控制器 (位置 / PD 力矩 / OSC)
- interfaces:   通信抽象层 RobotInterface + MuJoCo 实现
- envs:         Gymnasium 强化学习环境
"""

__version__ = "0.1.0"
