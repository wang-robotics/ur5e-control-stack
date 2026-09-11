"""演示: 笛卡尔直线 (LINE) 轨迹 —— TCP 在工作平面画一个正方形。

用法:
    python scripts/demo_traj.py            # 弹出 MuJoCo viewer 可视化
    python scripts/demo_traj.py --headless # 无渲染快速验证
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# 允许从任意目录运行: 项目根加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_ctrl import config
from arm_ctrl.interfaces import MuJoCoInterface


def main() -> None:
    parser = argparse.ArgumentParser(description="UR5e 直线轨迹演示")
    parser.add_argument(
        "--headless", action="store_true", help="不启动 viewer"
    )
    args = parser.parse_args()

    iface = MuJoCoInterface()
    iface.connect()

    viewer = None
    if not args.headless:
        import mujoco.viewer  # noqa: F401  显式导入 viewer 子模块

        viewer = mujoco.viewer.launch_passive(iface.model, iface.data)

    # 正方形顶点: 以当前 TCP 的 xy 为中心, 固定工作高度 z=0.30
    pos, _ = iface.get_tcp_pose()
    x0, y0 = float(pos[0]), float(pos[1])
    side = 0.12
    corners = [
        np.array([x0, y0, 0.30]),
        np.array([x0 + side, y0, 0.30]),
        np.array([x0 + side, y0 + side, 0.30]),
        np.array([x0, y0 + side, 0.30]),
        np.array([x0, y0, 0.30]),  # 闭合
    ]

    print(f"起点 TCP = {np.round(pos, 3)}")
    for i, corner in enumerate(corners):
        print(f"[LINE {i + 1}/{len(corners)}] -> {np.round(corner, 3)}")
        iface.move_linear(corner, duration=2.0, blocking=False)
        while iface.is_moving():
            iface.step()
            if viewer is not None:
                viewer.sync()
                time.sleep(config.CTRL_DT)
        p, _ = iface.get_tcp_pose()
        print(f"    到达: {np.round(p, 3)}")

    # 回 home
    print("[PTP] 返回 home")
    iface.move_joints(config.HOME_QP0, duration=2.5, blocking=False)
    while iface.is_moving():
        iface.step()
        if viewer is not None:
            viewer.sync()
            time.sleep(config.CTRL_DT)

    iface.disconnect()


if __name__ == "__main__":
    main()
