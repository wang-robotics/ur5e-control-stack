"""教学实验: 两连杆机械臂的正运动学 (FK) 手推 + 验算 (W2 Day 2)。

机械臂: 两根连杆, 长度 L1=L2=1, 都在 x-y 平面里摆动。
关节1 绕 z 轴转 θ1, 关节2 绕 z 轴转 θ2。

运行:
    python scripts/lesson_fk2d.py

输出:
    1. 三个 4x4 矩阵 (每个 = 一个连杆的"转+挪")
    2. 连乘 T = T01 @ T12 @ T2E 得到手的位置
    3. 与公式手算对照 (L1*c1+L2*c12, L1*s1+L2*s12)
    4. 肘和手两个点的坐标 (在纸上画出臂的形状)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

L1, L2 = 1.0, 1.0
DEG1, DEG2 = 30.0, 45.0


def T_rz_trans(deg: float, px: float, py: float, pz: float) -> np.ndarray:
    """绕 z 转 deg 度 + 平移 (px,py,pz) 的 4x4 齐次变换。"""
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [px, py, pz]
    return T


print("=" * 56)
print(f"两连杆臂: L1={L1}, L2={L2}, 关节角 {DEG1}° 和 {DEG2}°")
print("=" * 56)

T01 = T_rz_trans(DEG1, 0, 0, 0)    # 关节1: 转 θ1, 不挪 (在原点)
T12 = T_rz_trans(DEG2, L1, 0, 0)   # 关节2: 转 θ2, 再沿连杆1挪 L1
T2E = T_rz_trans(0.0, L2, 0, 0)    # 末端: 不转, 沿连杆2挪 L2

print("T01 (关节1):")
print(np.round(T01, 3))
print("T12 (关节2):")
print(np.round(T12, 3))

# 连乘: 右边先执行 (T2E 先, T01 最后)
T_hand = T01 @ T12 @ T2E
elbow = T01 @ T12 @ np.array([0.0, 0.0, 0.0, 1.0])
hand = T_hand @ np.array([0.0, 0.0, 0.0, 1.0])

print("肘的位置:", np.round(elbow[:3], 4))
print("手的位置:", np.round(hand[:3], 4))

# 公式手算对照: x = L1*cos(θ1) + L2*cos(θ1+θ2), y = L1*sin(θ1) + L2*sin(θ1+θ2)
t1 = np.deg2rad(DEG1)
t2 = np.deg2rad(DEG2)
x_formula = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
y_formula = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
print()
print("公式手算: (", round(x_formula, 4), ",", round(y_formula, 4), ", 0 )")
print("矩阵推导: (", round(hand[0], 4), ",", round(hand[1], 4), ", 0 )")
print("两者一致 -> 矩阵连乘 = 公式手算")
