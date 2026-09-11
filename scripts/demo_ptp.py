"""演示: 关节空间点到点 (PTP) 运动。

用法:
    python scripts/demo_ptp.py            # 弹出 MuJoCo viewer 可视化
    python scripts/demo_ptp.py --headless # 无渲染快速验证
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
    parser = argparse.ArgumentParser(description="UR5e PTP 演示")
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

    # 目标位姿序列 (以 home 为起点)
    q_targets = [
        np.array([1.2, -1.0, 0.8, -0.5, 0.5, 0.0]),
        np.array([-1.0, -0.6, 1.2, -1.0, 0.8, 1.0]),
        np.array([0.3, -1.5, -0.4, 0.2, -0.6, -1.5]),
    ]

    print(f"home = {np.round(iface.get_joint_positions(), 3)}")
    for i, q_goal in enumerate(q_targets):
        print(f"[PTP {i + 1}/{len(q_targets)}] -> {np.round(q_goal, 3)}")
        iface.move_joints(q_goal, duration=2.5, blocking=False)
        while iface.is_moving():
            iface.step()
            if viewer is not None:
                viewer.sync()
                time.sleep(config.CTRL_DT)
        print(f"    到达: {np.round(iface.get_joint_positions(), 3)}")

    # 回 home
    print("[PTP] 返回 home")
    iface.move_joints(config.HOME_QP0, duration=2.5, blocking=False)
    while iface.is_moving():
        iface.step()
        if viewer is not None:
            viewer.sync()
            time.sleep(config.CTRL_DT)

    print(f"final = {np.round(iface.get_joint_positions(), 3)}")
    iface.disconnect()


if __name__ == "__main__":
    main()
