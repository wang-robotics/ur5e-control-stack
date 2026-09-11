"""最小演示: 一串代码让机械臂动起来 (教学用)。

你的直觉没错 —— 如果只是"让它去几个固定位置", 一个文件就够了:
   1. 加载模型
   2. 把 6 个关节目标角写进 ctrl
   3. 反复推进仿真 + 渲染

项目里其他文件 (IK / 轨迹 / 接口 / RL) 都是在这个"最小内核"上,
为解决更多问题一层层加出来的。对比着看就能理解"为什么要分文件"。

用法:
    python scripts/demo_minimal.py
    python scripts/demo_minimal.py --headless   # 无窗口纯计算
"""
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

# ---- 1. 加载模型 (项目内置的 UR5e 3D 模型) ----
XML_PATH = (
    Path(__file__).resolve().parent.parent
    / "models" / "ur5e" / "universal_robots_ur5e" / "scene.xml"
)
model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

# 让内置伺服响应快一点 (原模型偏慢, 只影响动作快慢)
model.actuator_biasprm[:, 2] = -0.05 * model.actuator_gainprm[:, 0]

# ---- 2. 想去的 3 个位置: 每个位置 = 6 个关节要转的角度 (弧度) ----
HOME = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
targets = [
    np.array([1.2, -1.0, 0.8, -0.5, 0.5, 0.0]),
    np.array([-1.0, -0.6, 1.2, -1.0, 0.8, 1.0]),
    np.array([0.3, -1.5, -0.4, 0.2, -0.6, -1.5]),
    HOME,
]
data.qpos[:6] = HOME
data.ctrl[:] = HOME

# ---- 3. 依次去每个位置: 下发命令 + 推进仿真 2.5 秒 ----
viewer = None
if "--headless" not in sys.argv:
    import mujoco.viewer  # noqa: F401  viewer 需要显式导入

    viewer = mujoco.viewer.launch_passive(model, data)

for i, target in enumerate(targets):
    print(f"去位置 {i + 1}/{len(targets)}: {np.round(target, 2)}")
    data.ctrl[:] = target
    for step in range(1250):  # 1250 步 x 0.002s = 2.5 秒仿真时间
        mujoco.mj_step(model, data)
        if viewer is not None and step % 10 == 0:  # 每 10 步渲染一帧
            viewer.sync()
            time.sleep(0.02)

print("完成!")
