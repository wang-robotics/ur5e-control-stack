"""VLA 数据采集: 遥操作 UR5e + 记录 (图像, 动作) 对。

显示: MuJoCo 原生 3D 窗口 (鼠标拖拽转视角, 拖边角调大小)
输入: Win32 全局键态 (不管哪个窗口有焦点都能读到, 按住动/松开停)

用法:
    python scripts/teleop_collect.py --name ep1     # 遥操作
    python scripts/teleop_collect.py --selftest     # 无窗口自测

按键 (按住持续动, 松开停):
    W/S = 手前后(x)   A/D = 手左右(y)   Q/E = 手升降(z)
    J/L = 手绕 z 旋转    T = 结束保存    Esc = 放弃
    到达红球后停留 1.5 秒 -> 自动保存

注意: MuJoCo 窗口获得焦点时, 某些字母会切换它的渲染样式 (画面变样),
      不影响机械臂响应, 不必理会; 转视角用鼠标拖拽, 调大小拖窗口边角。

保存: datasets/teleop/<名字>.npz
      { images: (N,224,224,3), actions: (N,6), goal: (3,) }
"""
import argparse
import ctypes
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer  # noqa: F401  viewer 需显式导入
import numpy as np

# Win32 全局按键"按住"状态 (游戏式)
_VK = {"w": 0x57, "s": 0x53, "a": 0x41, "d": 0x44,
       "q": 0x51, "e": 0x45, "j": 0x4A, "l": 0x4C,
       "t": 0x54, "esc": 0x1B}


def key_held(name: str) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(_VK[name]) & 0x8000)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_ctrl import config
from arm_ctrl.interfaces import MuJoCoInterface
from arm_ctrl.robot import Robot

SCENE = (
    Path(__file__).resolve().parent.parent
    / "models" / "ur5e" / "universal_robots_ur5e" / "teleop_scene.xml"
)
IMG_SIZE = 224          # 训练用小图 (录制)
LIN_V = 0.15            # 键盘线速度 (m/s)
ANG_V = 0.8             # 键盘角速度 (rad/s)
SAMPLE_EVERY = 2        # 每 2 个控制周期采 1 帧 -> 25 Hz
MAX_DUR = 90.0          # 单次采集最长秒数
SUCCESS_DIST = 0.09     # 手离球心小于此距离视为"贴住" (球半径 6cm, 虚影可穿过)
SUCCESS_TICKS = 100     # 停稳(松键)后持续步数 (100 x 0.02s = 2 秒)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ep1", help="保存的文件名")
    parser.add_argument("--out", default="datasets/teleop")
    parser.add_argument("--selftest", action="store_true", help="无窗口自测")
    args = parser.parse_args()

    robot = Robot(xml_path=str(SCENE))
    iface = MuJoCoInterface(robot=robot)
    iface.connect()

    # 随机放目标球: 手前方 12~22cm (更近, 更容易一气呵成)
    rng = np.random.default_rng()
    tcp0, _ = iface.get_tcp_pose()
    goal = tcp0 + np.array(
        [rng.uniform(-0.05, 0.05), rng.uniform(-0.22, -0.12), rng.uniform(-0.06, 0.06)]
    )
    goal[2] = max(goal[2], 0.15)
    robot.model.body("target_ball").pos = goal
    mujoco.mj_forward(robot.model, robot.data)

    renderer = mujoco.Renderer(robot.model, IMG_SIZE, IMG_SIZE)
    images: list = []
    actions: list = []

    viewer = None
    if not args.selftest:
        viewer = mujoco.viewer.launch_passive(robot.model, robot.data)

    twist = np.zeros(6)
    t = 0.0
    step = 0
    done = False
    aborted = False
    was_moving = False
    near_ticks = 0

    print(f"目标球: {np.round(goal, 3)}")
    print("按键: W/S前后 A/D左右 Q/E升降 J/L旋转  T保存  Esc放弃")
    print("转视角: 鼠标拖拽 3D 窗口; 调大小: 拖窗口边角")
    try:
        while not done and t < MAX_DUR:
            if args.selftest:
                twist = np.array([0.05, 0.0, -0.04, 0.0, 0.0, 0.1])
                done = t > 3.0
            else:
                twist[:] = 0.0
                if key_held("w"):
                    twist[0] += LIN_V
                if key_held("s"):
                    twist[0] -= LIN_V
                if key_held("d"):
                    twist[1] += LIN_V
                if key_held("a"):
                    twist[1] -= LIN_V
                if key_held("e"):
                    twist[2] += LIN_V
                if key_held("q"):
                    twist[2] -= LIN_V
                if key_held("j"):
                    twist[5] += ANG_V
                if key_held("l"):
                    twist[5] -= ANG_V
                if key_held("t"):
                    done = True
                if key_held("esc"):
                    aborted = True
                    done = True

            moving = bool(np.any(twist != 0.0))
            if moving:
                iface.set_tcp_velocity(twist)
            elif was_moving:
                iface.stop()  # 刚松键: 锁一次目标, 之后保持
            was_moving = moving
            iface.step()

            # 到达判定: 手在球 9cm 内 且 完全松键(停稳) 持续 2 秒 -> 自动保存
            # 关键: 只要还在按键微调, 计数就清零 —— 存的是"停稳后的姿态"
            hand, _ = iface.get_tcp_pose()
            dist = float(np.linalg.norm(goal - hand))
            if dist < SUCCESS_DIST and not moving:
                near_ticks += 1
                if near_ticks >= SUCCESS_TICKS:
                    print("已停稳到达, 自动保存!")
                    done = True
            else:
                near_ticks = 0

            if step % 25 == 0:  # 每秒报一次距离
                state = "停稳中..." if (dist < SUCCESS_DIST and not moving) else ("贴住了, 松键停稳!" if dist < SUCCESS_DIST else "")
                print(f"  [{t:5.1f}s] 手离球 {dist*100:5.1f} cm  {state}")

            if step % SAMPLE_EVERY == 0:
                renderer.update_scene(robot.data)
                images.append(renderer.render().copy())
                actions.append(twist.copy())
            if viewer is not None:
                viewer.sync()
                time.sleep(0.004)
            step += 1
            t += config.CTRL_DT
    except Exception:
        import traceback

        traceback.print_exc()
        print("发生错误, 数据未保存。请把上面的错误信息发给 AI。")
        try:
            input("按回车退出...")
        except Exception:
            pass
        return

    if viewer is not None:
        viewer.close()

    if aborted:
        print("已放弃, 未保存。")
        return

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{args.name}.npz"
    np.savez_compressed(
        out_path,
        images=np.stack(images).astype(np.uint8),
        actions=np.stack(actions).astype(np.float32),
        goal=goal.astype(np.float32),
    )
    hand, _ = iface.get_tcp_pose()
    print(f"已保存 {len(images)} 个样本 -> {out_path}")
    print(f"最终手位置 {np.round(hand, 3)}, 离目标 {np.linalg.norm(goal - hand):.3f} m")


if __name__ == "__main__":
    main()
