"""定位 ep20 回放分歧: 对比 积分 / FK(q_des) / 物理实际手 三条轨迹。"""
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

robot = Robot(xml_path=str(SCENE))
iface = MuJoCoInterface(robot=robot)
iface.connect()
tcp0, _ = iface.get_tcp_pose()
robot.model.body("target_ball").pos = goal
mujoco.mj_forward(robot.model, robot.data)
iface.reset()
mujoco.mj_forward(robot.model, robot.data)

site_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

def fk(q):
    tmp = mujoco.MjData(robot.model)
    tmp.qpos[: robot.model.nq] = q
    mujoco.mj_kinematics(robot.model, tmp)
    return tmp.site_xpos[site_id].copy()

vel = actions[:, :3].astype(float)
traj_int = tcp0 + np.cumsum(vel, axis=0) * config.CTRL_DT * 2

prev = np.zeros(6)
rows = []
for i, a in enumerate(actions):
    a = a.astype(float)
    moving = bool(np.any(a != 0.0))
    was_moving = bool(np.any(prev != 0.0))
    for _ in range(2):
        if moving:
            iface.set_tcp_velocity(a)
        elif was_moving:
            iface.stop()
        iface.step()
    prev = a
    if i % 100 == 0 or i == len(actions) - 1:
        hand, _ = iface.get_tcp_pose()
        q_des = iface._osc_ctrl.q_des.copy()
        fk_qdes = fk(q_des)
        rows.append((i, hand, fk_qdes))

print(f"{'帧':>5} | {'物理手距球':>9} | {'FK(q_des)距球':>11} | {'积分距球':>8} | 物理手位置 | FK(q_des)位置")
for i, hand, fk_qdes in rows:
    dh = np.linalg.norm(goal - hand) * 100
    df = np.linalg.norm(goal - fk_qdes) * 100
    di = np.linalg.norm(goal - traj_int[i]) * 100
    print(f"{i:5d} | {dh:8.2f}cm | {df:10.2f}cm | {di:7.2f}cm | "
          f"{np.round(hand,2)} | {np.round(fk_qdes,2)}")

iface.disconnect()
