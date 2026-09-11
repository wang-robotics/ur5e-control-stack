"""验证: 正确的保持逻辑 (松键时锁一次目标) 下, 各增益倍率的稳定性。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_ctrl.controller import set_position_servo
from arm_ctrl.interfaces import MuJoCoInterface
from arm_ctrl.robot import Robot

SCENE = (
    Path(__file__).resolve().parent.parent
    / "models" / "ur5e" / "universal_robots_ur5e" / "teleop_scene.xml"
)

for mult in [1.0, 2.0, 3.0]:
    robot = Robot(xml_path=str(SCENE))
    iface = MuJoCoInterface(robot=robot)
    robot.model.actuator_gainprm[:, 0] *= mult
    set_position_servo(robot.model, kv_ratio=0.05)
    iface.connect()
    iface.stop()  # 只锁一次目标 (正确逻辑)
    q0 = iface.get_joint_positions()
    for _ in range(150):  # 3 秒
        iface.step()
    q1 = iface.get_joint_positions()
    drift = float(np.max(np.abs(q1 - q0)))
    print(f"倍率 x{mult}: 3秒最大漂移 = {np.rad2deg(drift):.2f}°  {'稳定' if drift < 0.03 else '失稳!'}")
