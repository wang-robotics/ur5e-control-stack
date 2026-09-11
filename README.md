# arm_ctrl —— MuJoCo 六轴机械臂 (UR5e) 控制栈

> **作者**: 王智伟 (Wang Zhiwei) · 广东工业大学 机械设计制造及其自动化 (2022 级) ·
> 2791842174@qq.com
>
> **项目主页**: https://github.com/wang-robotics/ur5e-control-stack
>
> 实验结果与英文版说明见 [`README_EN.md`](README_EN.md); 完整实验报告见
> [`docs/实验报告.md`](docs/实验报告.md)。

基于 **MuJoCo + Python** 的六轴机械臂完整控制栈，面向具身智能（强化学习）方向。
包含：正/逆运动学、轨迹规划（PTP/LINE）、关节控制（位置/PD 力矩/OSC）、
通信抽象接口、Gymnasium 强化学习环境与 SB3 训练脚本。

## 架构

```
┌────────────────────────────────────────────────────┐
│  上层应用                                            │
│  RL 环境 (envs/ur5e_reach.py)   演示脚本 (scripts/)  │
├────────────────────────────────────────────────────┤
│  RobotInterface (interfaces.py)  ← 通信抽象层 ★      │
│  轨迹级: move_joints / move_linear                   │
│  流控级: set_joint_target / set_tcp_velocity         │
├────────────────────────────────────────────────────┤
│  控制器 (controller.py)  位置 / PD力矩 / OSC         │
│  轨迹规划 (trajectory.py)  PTP / LINE               │
│  运动学 (robot.py)  FK / IK(DLS) / Jacobian         │
├────────────────────────────────────────────────────┤
│  MuJoCo 仿真 (Menagerie UR5e 模型)                   │
└────────────────────────────────────────────────────┘
```

★ 关键设计：上层只依赖 `RobotInterface` 抽象，不接触 MuJoCo API。
将来接入真机（ROS2 / gRPC / 串口）时只需新增一个接口实现类，控制栈与
RL 环境零改动。

## 模型权重与演示视频

训练好的策略权重与演示片段见 **[Release v1.0](https://github.com/wang-robotics/ur5e-control-stack/releases/tag/v1.0)**：

| 内容 | 文件 |
|---|---|
| PPO 策略（原模型 + 3 个种子） | `ur5e_reach_ppo*.zip` |
| 行为克隆策略（主模型 / v3 / 同配方重训 / 干净数据对照） | `vla_bc*.pt` |
| 演示片段（6 段，480×480，25 fps） | `ppo_v2_ep0-2.mp4`、`vla_v2_ep0-2.mp4` |

> 演示片段与上面的数字**用同一套评估协议**渲染（同种子、无动作平滑、达到成功判据即停），
> 片段末尾的落点就是评估日志里记录的最终距离。因此 **VLA 片段末尾仍留有数厘米间隙** ——
> 它停在 **9 cm 成功判据**处（9 cm 是报告中对行为克隆使用的口径），这与报告数字一致，不是渲染问题。

> 数据集（约 1 GB）不进仓库：用 `scripts/teleop_collect.py` + `scripts/preprocess_vla.py`
> 复现，说明见 `data/README.md`。

## 环境准备

推荐使用 conda 环境（本机已有 `mujoco_project` 可直接使用）：

```powershell
conda activate mujoco_project
pip install -r requirements.txt
```

## 快速开始

```powershell
conda activate mujoco_project

# 演示: 关节空间点到点 (弹出 viewer 可视化)
python scripts/demo_ptp.py

# 演示: 笛卡尔直线画正方形
python scripts/demo_traj.py

# 无渲染快速验证 (服务器/CI)
python scripts/demo_ptp.py --headless

# 单元测试
python -m unittest discover -s tests -v

# RL 训练 (PPO, 末端空间动作)
python scripts/train_sb3.py --timesteps 200000
# 评估
python scripts/train_sb3.py --eval
```

## 模块说明

| 模块 | 职责 |
|---|---|
| `arm_ctrl/config.py` | 关节限位、控制频率、默认位姿、奖励参数等全局配置 |
| `arm_ctrl/robot.py` | 模型加载、FK（mj_kinematics 直读 site）、IK（自研阻尼最小二乘迭代）、数值雅可比 |
| `arm_ctrl/trajectory.py` | PTP（关节空间 quintic）、LINE（笛卡尔直线 + SLERP + 逐点 IK） |
| `arm_ctrl/controller.py` | `JointPositionController` / `JointTorquePDController` / `OSCController` |
| `arm_ctrl/interfaces.py` | `RobotInterface` 抽象基类 + `MuJoCoInterface` 实现 |
| `arm_ctrl/envs/ur5e_reach.py` | Gymnasium Reach 环境（末端空间动作） |
| `scripts/` | 演示与训练脚本 |
| `tests/` | unittest 单元测试 |

## 模型来源

UR5e 模型（官方 Menagerie 版）已随项目内置在 `models/ur5e/`，包含全部
OBJ 网格文件，开箱即用。如需重新下载/更新：

```powershell
python scripts/download_ur5e_model.py
```

## 实现说明

- **IK**：自研阻尼最小二乘（DLS）迭代 `dq = J^T(JJ^T + λ²I)⁻¹err`，
  基于中心差分数值雅可比，带关节步长裁剪与限位，对奇异位形鲁棒、
  零外部依赖（注：mujoco 3.9 已无内置 IK；mink 1.2 与 mujoco 3.9
  存在求解器兼容问题，故采用自研方案）
- **关节控制**：默认使用模型内建位置伺服（ctrl = 目标角），并将阻尼比
  kv/kp 从模型原值 0.2 调至 0.05（带宽 0.8Hz → ~20Hz，匹配 50Hz 控制
  周期）；另提供力矩模式 PD + 重力补偿（`enable_torque_mode` +
  `JointTorquePDController`）
- **OSC**：末端 6 维速度经 DLS 映射为关节速度并积分成位置目标，
  即 RL 的 task-space 动作入口

## RL 环境 (UR5eReachEnv)

- **任务**：控制 TCP 到达工作空间内随机目标点（2 cm 内视为成功）
- **动作**（6 维，末端空间）：归一化末端速度 × `[0.15 m/s]×3 + [0.8 rad/s]×3`，
  内部经 OSC（阻尼最小二乘差分 IK）映射为关节命令
- **观测**（21 维）：`qpos(6) + qvel(6) + tcp_pos(3) + goal_pos(3) + goal_rel(3)`
- **奖励**：距离减少塑形 + 指数接近奖励 + 动作/关节速度惩罚 + 成功奖励
- **步长**：控制周期 20 ms（50 Hz），内部 2 ms 物理子步

## 单位与坐标约定

- 位置 `m`，姿态四元数 `wxyz`（MuJoCo 原生约定）
- 关节角 `rad`，世界坐标系即 MuJoCo 场景坐标系（UR5e 基座位于原点）
