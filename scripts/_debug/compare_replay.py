"""对比 ep20 的两种重建: (a) 运动学积分 vs (b) 物理回放。"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from arm_ctrl import config
from arm_ctrl.interfaces import MuJoCoInterface
from arm_ctrl.robot import Robot

SCENE = (
    Path(__file__).resolve().parent.parent.parent
    / "models" / "ur5e" / "universal_robots_ur5e" / "teleop_scene.xml"
)
EP = Path("datasets/teleop/ep20.npz")

d = np.load(EP)
actions = d["actions"]
goal = d["goal"].astype(float)

# (a) 运动学积分
robot_a = Robot(xml_path=str(SCENE))
iface_a = MuJoCoInterface(robot=robot_a)
iface_a.connect()
tcp0, _ = iface_a.get_tcp_pose()
vel = actions[:, :3].astype(float)
disp = np.cumsum(vel, axis=0) * config.CTRL_DT * 2
traj_a = tcp0 + disp
dist_a = np.linalg.norm(goal - traj_a, axis=1)

# (b) 物理回放
robot_b = Robot(xml_path=str(SCENE))
iface_b = MuJoCoInterface(robot=robot_b)
iface_b.connect()
robot_b.model.body("target_ball").pos = goal
mujoco.mj_forward(robot_b.model, robot_b.data)
iface_b.reset()
mujoco.mj_forward(robot_b.model, robot_b.data)

traj_b = []
dist_b = []
prev = np.zeros(6)
for i, a in enumerate(actions):
    a = a.astype(float)
    moving = bool(np.any(a != 0.0))
    was_moving = bool(np.any(prev != 0.0))
    for _ in range(2):
        if moving:
            iface_b.set_tcp_velocity(a)
        elif was_moving:
            iface_b.stop()
        iface_b.step()
        hand, _ = iface_b.get_tcp_pose()
        traj_b.append(hand.copy())
        dist_b.append(float(np.linalg.norm(goal - hand)))
    prev = a
traj_b = np.array(traj_b)
dist_b = np.array(dist_b)

print(f"帧数 {len(actions)}, 积分终点 {dist_a[-1]*100:.2f}cm, 回放终点 {dist_b[-1]*100:.2f}cm")
print(f"积分轨迹最近 {dist_a.min()*100:.2f}cm, 回放轨迹最近 {dist_b.min()*100:.2f}cm")
print("\n按 10% 进度对比 (每行: 帧序号 | 积分距离 | 回放距离 | 差):")
for k in range(0, len(actions), max(1, len(actions) // 12)):
    i = k  # 采样帧索引
    j = 2 * k  # 回放步索引
    print(f"  f{i:4d}: int {dist_a[i]*100:6.2f} | sim {dist_b[j]*100:6.2f} | Δ {abs(dist_a[i]-dist_b[j])*100:6.2f} cm")

# 最大分歧点
diffs = np.abs(dist_a - dist_b[::2][: len(dist_a)])
k = int(np.argmax(diffs))
print(f"\n最大分歧 f{k}: int {dist_a[k]*100:.2f} vs sim {dist_b[2*k]*100:.2f}")
print(f"该帧动作: {np.round(actions[k], 3)}")
print(f"积分手位置: {np.round(traj_a[k], 3)}")
print(f"回放手位置: {np.round(traj_b[2*k], 3)}")
iface_a.disconnect()
iface_b.disconnect()
