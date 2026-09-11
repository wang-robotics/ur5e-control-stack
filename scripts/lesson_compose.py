"""教学实验: 矩阵连乘 —— 顺序铁律 (W2 进阶)。

运行:
    python scripts/lesson_compose.py

核心问题: 先绕z转90、再绕x转90, 和反过来做, 结果一样吗?
答案: 不一样! 矩阵乘法不满足交换律, 这决定了 FK 推导里的乘法顺序。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def Rz(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    return np.array(
        [
            [np.cos(t), -np.sin(t), 0.0],
            [np.sin(t), np.cos(t), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def Rx(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(t), -np.sin(t)],
            [0.0, np.sin(t), np.cos(t)],
        ]
    )


p = np.array([1.0, 0.0, 0.0])  # 初始点: x 轴上的 (1,0,0)

print("=" * 56)
print("实验: 同一个点 (1,0,0), 两种顺序各转两次")
print("=" * 56)

# 顺序A: 先绕 z 转 90, 再绕 x 转 90
order_A = Rx(90) @ Rz(90)
# 顺序B: 先绕 x 转 90, 再绕 z 转 90
order_B = Rz(90) @ Rx(90)

print("顺序A  Rx(90) @ Rz(90) @ p =", np.round(order_A @ p, 6))
print("顺序B  Rz(90) @ Rx(90) @ p =", np.round(order_B @ p, 6))
print()
print("结果不同! A 到了 z 轴, B 到了 y 轴。")
print()
print("铁律: 写在【右边】的矩阵【先执行】。")
print("  Rx(90) @ Rz(90) @ p = Rx(90) @ ( Rz(90) @ p )   <- Rz 先转")
print("  Rz(90) @ Rx(90) @ p = Rz(90) @ ( Rx(90) @ p )   <- Rx 先转")
print()
print("像穿衣服: 先穿(右)的在里面, 后穿(左)的在外面。")
