"""操控练习场: 4 种方式操控机械臂 (教学用)。

这是从"最小脚本"走向"完整控制栈"的桥:
- demo_minimal.py 直接写底层 ctrl, 一个文件搞定
- 本脚本通过 RobotInterface 操控, 展示项目提供的 4 种操控层级

每种方式跑一遍, 感受"层级越高越省心, 层级越低越直接"。

用法 (用数字选方式, 不加数字默认方式1):
    python scripts/playground.py 1    # 方式1: 给目标关节角, 自动规划轨迹
    python scripts/playground.py 2    # 方式2: 流控, 直接覆盖关节目标角
    python scripts/playground.py 3    # 方式3: 只给手的位置, IK 自动解算
    python scripts/playground.py 4    # 方式4: 给手一个速度 (AI 训练同款)
    任意方式后加 --headless 可不弹窗
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# 允许从任意目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_ctrl import config
from arm_ctrl.interfaces import MuJoCoInterface


def make_viewer(iface: MuJoCoInterface, headless: bool):
    if headless:
        return None
    import mujoco.viewer  # noqa: F401  viewer 需要显式导入

    return mujoco.viewer.launch_passive(iface.model, iface.data)


def wait_viewer(viewer) -> None:
    if viewer is not None:
        viewer.sync()
        time.sleep(config.CTRL_DT)


# --------------------------------------------------------------------------
def demo_ptp(iface: MuJoCoInterface, viewer) -> None:
    """方式1: move_joints —— 告诉它 6 个关节的目标角, 内部自动规划平滑轨迹。

    最常用、最省心: 不用管中间过程, 保证起停平稳。
    """
    print("== 方式1: move_joints (关节空间点到点) ==")
    targets = [
        config.HOME_QP0 + np.array([0.8, -0.5, 0.5, 0.3, -0.5, 0.5]),
        config.HOME_QP0 + np.array([-0.8, 0.3, -0.5, -0.4, 0.5, -0.8]),
        config.HOME_QP0,
    ]
    for q_goal in targets:
        print("  目标关节角:", np.round(q_goal, 2))
        iface.move_joints(q_goal, duration=2.0, blocking=False)
        while iface.is_moving():
            iface.step()
            wait_viewer(viewer)
        print("  实际关节角:", np.round(iface.get_joint_positions(), 2))


def demo_joint_streaming(iface: MuJoCoInterface, viewer) -> None:
    """方式2: set_joint_target —— 每个控制周期直接覆盖目标角, 不做轨迹规划。

    适合: 键盘/手柄遥操作, 或外部算法每周期下发命令。
    这里演示只动第一个关节 (底座), 正弦摆动。
    """
    print("== 方式2: set_joint_target (关节位置流控) ==")
    q0 = config.HOME_QP0.copy()
    for k in range(80):
        q = q0.copy()
        q[0] = q0[0] + 0.6 * np.sin(k * 0.05)  # 只有关节 0 在摆
        iface.set_joint_target(q)
        iface.step()
        if k % 20 == 0:
            print(
                "  关节0 目标:", round(q[0], 2),
                " 实际:", round(iface.get_joint_positions()[0], 2),
            )
        wait_viewer(viewer)


def demo_line(iface: MuJoCoInterface, viewer) -> None:
    """方式3: move_linear —— 只告诉它"手"要去哪 (空间坐标), IK 自动解关节角。

    你不用想关节怎么转 —— 这是最贴近人类直觉的操控方式。
    """
    print("== 方式3: move_linear (笛卡尔直线) ==")
    p_home, _ = iface.get_tcp_pose()
    p_low = p_home.copy()
    p_low[2] = 0.30  # 手向下到 0.30m 高度
    p_side = p_low.copy()
    p_side[0] += 0.10  # 再水平挪 10cm
    for goal in [p_low, p_side, p_home]:
        print("  手的目标位置:", np.round(goal, 3))
        iface.move_linear(goal, duration=2.0, blocking=False)
        while iface.is_moving():
            iface.step()
            wait_viewer(viewer)
        p, _ = iface.get_tcp_pose()
        print("  手的实际位置:", np.round(p, 3))


def demo_osc(iface: MuJoCoInterface, viewer) -> None:
    """方式4: set_tcp_velocity —— 给"手"一个速度, OSC 实时转成关节命令。

    这是 AI 训练用的动作方式: 策略每周期输出"手往哪动、多快"。
    这里演示手沿 +x / -x 交替移动。
    """
    print("== 方式4: set_tcp_velocity (末端速度流控 / OSC) ==")
    for k in range(150):
        vx = 0.08 if (k // 75) % 2 == 0 else -0.08
        iface.set_tcp_velocity([vx, 0.0, 0.0, 0.0, 0.0, 0.0])
        iface.step()
        if k % 30 == 0:
            p, _ = iface.get_tcp_pose()
            print("  手的位置:", np.round(p, 3))
        wait_viewer(viewer)
    iface.stop()


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    headless = "--headless" in sys.argv
    mode = args[0] if args else "1"

    demos = {
        "1": demo_ptp,
        "2": demo_joint_streaming,
        "3": demo_line,
        "4": demo_osc,
    }
    if mode not in demos:
        print(f"未知方式 {mode}, 可选: {sorted(demos)}")
        sys.exit(1)

    iface = MuJoCoInterface()
    iface.connect()
    viewer = make_viewer(iface, headless)

    demos[mode](iface, viewer)

    print("演示结束。")
    iface.disconnect()


if __name__ == "__main__":
    main()
