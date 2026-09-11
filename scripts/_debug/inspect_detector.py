"""现场核查 eval_vla_detect 的检测器: 每步是否成功检测, 估计值与真值差多少。

跑 2 局, 每局打印: 检测成功步数/总步数, 以及前 5 次成功检测的 |估计-真值| 误差。
用于判断 v4.5≈v4 到底是 (a) 检测一直失败回退真值, 还是 (b) 近距离下检测本来就准。
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ 目录

from arm_ctrl.interfaces import MuJoCoInterface
from arm_ctrl.robot import Robot
from eval_vla_detect import RGBDBallDetector, SCENE

robot = Robot(xml_path=str(SCENE))
iface = MuJoCoInterface(robot=robot)
iface.connect()
renderer = mujoco.Renderer(robot.model, 224, 224)
detector = RGBDBallDetector(renderer)
rng = np.random.default_rng(42)

for ep in range(2):
    iface.reset()
    tcp0, _ = iface.get_tcp_pose()
    goal = tcp0 + np.array(
        [rng.uniform(-0.05, 0.05), rng.uniform(-0.22, -0.12), rng.uniform(-0.06, 0.06)]
    )
    goal[2] = max(goal[2], 0.15)
    robot.model.body("target_ball").pos = goal
    mujoco.mj_forward(robot.model, robot.data)

    n_ok = 0
    n_fail = 0
    errs: list[float] = []
    for step in range(20):  # 只看前 20 步 (开局段)
        est = detector.detect(robot.data)
        if est is None:
            n_fail += 1
        else:
            n_ok += 1
            if len(errs) < 5:
                errs.append(float(np.linalg.norm(est - goal) * 100))
        iface.set_tcp_velocity(np.zeros(6))
        iface.step()
    print(
        f"ep{ep}: 前20步 检测成功 {n_ok} 次, 失败 {n_fail} 次 | "
        f"前几次误差(cm): {[f'{e:.1f}' for e in errs]}"
    )
    print(f"  球真值: {np.round(goal, 3)}")

iface.disconnect()
