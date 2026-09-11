"""回放 datasets/teleop/*.npz 的遥操作示教, 计算每条示教的终止距离 (手-球心)。

回放语义与 teleop_collect.py 完全一致:
    - 每个存储动作连续施加 2 个控制周期 (SAMPLE_EVERY=2)
    - 动作从非零变零的那一步调用一次 stop() (锁目标), 之后空步
    - 初始位姿 = 场景 keyframe (采集时同样如此)
输出: 每条示教 |terminal - goal| 的均值±标准差, 用于验证"示教终止 3.6±1.1cm"。
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from arm_ctrl.interfaces import MuJoCoInterface
from arm_ctrl.robot import Robot

SCENE = (
    Path(__file__).resolve().parent.parent.parent
    / "models" / "ur5e" / "universal_robots_ur5e" / "teleop_scene.xml"
)
TELEOP_DIR = Path(__file__).resolve().parent.parent.parent / "datasets" / "teleop"

robot = Robot(xml_path=str(SCENE))
iface = MuJoCoInterface(robot=robot)
iface.connect()

files = sorted(TELEOP_DIR.glob("*.npz"))
print(f"共 {len(files)} 条示教")

dists = []
for f in files:
    d = np.load(f)
    actions = d["actions"]
    goal = d["goal"].astype(float)
    robot.model.body("target_ball").pos = goal
    mujoco.mj_forward(robot.model, robot.data)
    iface.reset()  # 回到 home (采集时的初始位姿)
    mujoco.mj_forward(robot.model, robot.data)

    prev = np.zeros(6)
    for a in actions:
        a = a.astype(float)
        moving = bool(np.any(a != 0.0))
        was_moving = bool(np.any(prev != 0.0))
        for _ in range(2):  # SAMPLE_EVERY=2: 一个存储动作覆盖 2 个控制周期
            if moving:
                iface.set_tcp_velocity(a)
            elif was_moving:
                iface.stop()  # 松键瞬间锁一次
            iface.step()
        prev = a

    hand, _ = iface.get_tcp_pose()
    dist = float(np.linalg.norm(goal - hand))
    dists.append(dist)
    print(f"{f.name}: {len(actions)} 样本, 终止距离 {dist*100:5.2f} cm")

dists = np.array(dists)
print(f"\n示教终止距离: {dists.mean()*100:.2f} ± {dists.std()*100:.2f} cm (n={len(dists)})")
print(f"最小 {dists.min()*100:.2f} cm, 最大 {dists.max()*100:.2f} cm")

iface.disconnect()
