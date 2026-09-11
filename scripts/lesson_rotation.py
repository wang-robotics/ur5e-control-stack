"""教学实验: 旋转矩阵 —— 运动学数学的第一块地基 (W2 Day 1)。

运行:
    python scripts/lesson_rotation.py

三段演示:
    1. 2D 旋转: 一个角度 -> 2x2 矩阵, 把向量转过去
    2. 3D 旋转: 3x3 矩阵, 三列就是转后的三个坐标轴
    3. 4x4 齐次变换: 旋转 + 平移合体, 一步把点搬到新位置
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def rotate2d(p: np.ndarray, deg: float) -> np.ndarray:
    """2D 旋转: 角度(度) -> 旋转矩阵 -> 作用在点上。"""
    t = np.deg2rad(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return R, R @ p


def rotate3d_z(deg: float) -> np.ndarray:
    """3D 绕 z 轴旋转矩阵。"""
    t = np.deg2rad(deg)
    return np.array(
        [
            [np.cos(t), -np.sin(t), 0.0],
            [np.sin(t), np.cos(t), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


print("=" * 56)
print("1. 2D 旋转: 把点 (1, 0) 转 90 度")
print("=" * 56)
R2, p2 = rotate2d(np.array([1.0, 0.0]), 90)
print("旋转矩阵 R =")
print(R2)
print(f"R @ (1,0) = {p2}   <- (1,0) 被转到 (0,1), 竖起来了!")

print()
print("=" * 56)
print("2. 3D 绕 z 轴转 90 度: 三列就是转后的三个轴")
print("=" * 56)
R3 = rotate3d_z(90)
print("R =")
print(R3)
print("第 1 列 = 原来的 x 轴被转到哪:", R3[:, 0], " (指向 +y)")
print("第 2 列 = 原来的 y 轴被转到哪:", R3[:, 1], " (指向 -x)")
print("第 3 列 = 原来的 z 轴被转到哪:", R3[:, 2], " (z 不动)")
print("验证旋转矩阵是'正交'的: R @ R.T 应该 = 单位阵")
print(np.round(R3 @ R3.T, 6))

print()
print("=" * 56)
print("3. 4x4 齐次变换: 旋转 + 平移合体")
print("=" * 56)
T = np.eye(4)
T[:3, :3] = R3          # 左上角: 旋转
T[:3, 3] = [1, 2, 3]    # 右上角: 平移 (1,2,3)
print("T =")
print(T)
p = np.array([1.0, 0.0, 0.0, 1.0])  # 点 (1,0,0), 末尾补 1
print(f"T @ (1,0,0,1) = {T @ p[:4]}   <- 先转 90 度再平移 (1,2,3)")
print("齐次坐标: 点的末尾补一个 1, 就能用一次矩阵乘法同时做旋转+平移")
