"""验证采集数据: 红球与机械臂是否在画面中。"""
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else r"datasets\teleop\selftest2.npz"
d = np.load(path)
img = d["images"][0].astype(int)
red = (img[:, :, 0] > 120) & (img[:, :, 0] - img[:, :, 1] > 40) & (img[:, :, 0] - img[:, :, 2] > 40)
blue = (img[:, :, 2] > 100) & (img[:, :, 2] - img[:, :, 0] > 30)
print(f"文件: {path}")
print(f"样本数: {len(d['images'])}, 图像尺寸: {d['images'].shape[1:]}, 动作维度: {d['actions'].shape[1]}")
print(f"首帧: 红球像素 {red.sum()}, 蓝臂像素 {blue.sum()}")
print(f"目标球位置: {np.round(d['goal'], 3)}")
