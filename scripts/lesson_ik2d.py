"""教学实验: 两连杆机械臂的逆运动学 (IK) 概念 (W2 Day 2 加餐)。

FK: 给角度 -> 算手的位置 (昨天推过)
IK: 给手的位置 -> 反求角度

运行:
    python scripts/lesson_ik2d.py

演示三件事:
    1. 目标 (1,1) 有两个解: 肘在上 (0,90) 和 肘在下 (90,-90)
    2. 两个解算回 FK, 都到 (1,1) —— 验证多解
    3. 目标 (3,0) 太远 (>L1+L2=2), 无解
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

L1, L2 = 1.0, 1.0


def fk_2d(theta1_deg: float, theta2_deg: float) -> np.ndarray:
    """昨天的 FK 公式: 角度 -> 手的位置。"""
    t1 = np.deg2rad(theta1_deg)
    t2 = np.deg2rad(theta2_deg)
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return np.array([x, y])


print("=" * 56)
print("IK 演示: L1=L2=1, 手要去 (1, 1)")
print("=" * 56)

# 解 A: 肘在上 (0, 90) —— 棍1平躺, 棍2竖直
solA = (0.0, 90.0)
# 解 B: 肘在下 (90, -90) —— 棍1竖立, 棍2折回平躺
solB = (90.0, -90.0)

print(f"解A: 角度 = {solA}  -> FK 算回手的位置 = {np.round(fk_2d(*solA), 4)}")
print(f"解B: 角度 = {solB}  -> FK 算回手的位置 = {np.round(fk_2d(*solB), 4)}")
print("同一个目标, 两个不同的姿势都能到 -> IK 的'多解'")

print()
print("=" * 56)
print("无解演示: 手要去 (3, 0)")
print("=" * 56)
print("目标距基座 3 米, 但手臂最长只能伸 L1+L2 = 2 米")
print("-> 无解! IK 必须诚实报告失败 (项目里 success=False)")
