"""VLA 数据质检: 扫描 datasets/teleop 下所有采集数据并打分。"""
import sys
from pathlib import Path

import numpy as np

DATADIR = Path("datasets/teleop")


def red_ratio(img: np.ndarray) -> float:
    img = img.astype(int)
    red = (img[:, :, 0] > 120) & (img[:, :, 0] - img[:, :, 1] > 40) & (img[:, :, 0] - img[:, :, 2] > 40)
    return float(red.mean())


files = sorted(DATADIR.glob("*.npz"))
files = [f for f in files if not f.name.startswith("selftest")]
if not files:
    print("datasets/teleop 下没有找到数据 (排除 selftest)")
    sys.exit(0)

print(f"{'文件':<18}{'样本':>5}{'时长s':>7}{'红球可见':>9}{'零动作%':>8}{'动作幅度':>8}{'初始距离':>8}{'最终距离':>8}  评价")
all_ok = True
for f in files:
    d = np.load(f)
    imgs, acts, goal = d["images"], d["actions"], d["goal"]
    n = len(imgs)
    dur = n / 25.0  # 25Hz 采样
    red = np.mean([red_ratio(imgs[i]) for i in range(0, n, max(1, n // 20))])
    zero_pct = 100.0 * float(np.mean(np.all(np.abs(acts) < 1e-6, axis=1)))
    act_mag = float(np.mean(np.linalg.norm(acts, axis=1)))
    # 初始/最终距离: 图像里没法直接拿手位置, 用动作积分粗略估计不可靠;
    # 这里用最后一个非零动作帧之前的手到球距离不可得, 改为打印动作统计
    d_initial = "  -"
    # 最终距离近似: 最后 1/4 动作均值幅度 (还在动=没够到)
    tail_mag = float(np.mean(np.linalg.norm(acts[-max(1, n // 4):], axis=1)))
    problems = []
    if n < 100:
        problems.append("太短")
    if red < 0.0001:
        problems.append("红球不可见")
    if zero_pct > 50:
        problems.append("发呆过半")
    if act_mag < 0.01:
        problems.append("几乎没动")
    if tail_mag > 0.02:
        problems.append("结尾还在动(没到?)")
    verdict = "OK" if not problems else "[!] " + ",".join(problems)
    if problems:
        all_ok = False
    print(f"{f.name:<18}{n:>5}{dur:>7.1f}{red*100:>8.2f}%{zero_pct:>7.1f}%{act_mag:>8.4f}{d_initial:>8}{tail_mag:>8.4f}  {verdict}")

print()
print("总体:", "全部合格" if all_ok else "有需要重录的条目, 见上面 [!] 列")
print("说明: 红球可见=画面中含红球的比例(>0.02%算好); 零动作%=发呆时间;")
print("      结尾还在动=可能没够到球就按了 T")
