"""扫描 v4.5 50 局评估的球位序列 (rng=42), 统计每局开局时检测器能否看见红球。

背景: inspect_detector.py 发现有些球位下红像素 < 5 → detect() 返回 None,
      eval_vla_detect.py 会回退到真值 goal (仿真兜底)。本脚本量化覆盖率。
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_ctrl.interfaces import MuJoCoInterface
from arm_ctrl.robot import Robot
from eval_vla_detect import RGBDBallDetector, SCENE

robot = Robot(xml_path=str(SCENE))
iface = MuJoCoInterface(robot=robot)
iface.connect()
renderer = mujoco.Renderer(robot.model, 224, 224)
detector = RGBDBallDetector(renderer)

# 复现 eval_vla_detect.py 的球位序列: rng(42), 每局 reset 后采样
rng = np.random.default_rng(42)
seen, blind = [], []
for ep in range(50):
    iface.reset()
    tcp0, _ = iface.get_tcp_pose()
    goal = tcp0 + np.array(
        [rng.uniform(-0.05, 0.05), rng.uniform(-0.22, -0.12), rng.uniform(-0.06, 0.06)]
    )
    goal[2] = max(goal[2], 0.15)
    robot.model.body("target_ball").pos = goal
    mujoco.mj_forward(robot.model, robot.data)
    est = detector.detect(robot.data)
    err = float(np.linalg.norm(est - goal) * 100) if est is not None else None
    if est is not None:
        seen.append(err)
    else:
        blind.append(ep + 1)
    tag = f"{err:4.1f} cm" if err is not None else "看不见!"
    print(f"ep{ep+1:2d} 球 {np.round(goal,3)} 检测误差 {tag}")

errs = np.array(seen)
print(f"\n检测成功 {len(seen)}/50 局 (覆盖率 {len(seen)/50*100:.0f}%)")
print(f"成功局的误差: {errs.mean():.2f} ± {errs.std():.2f} cm")
print(f"检测失败的局 (将回退真值): {blind}")

iface.disconnect()
