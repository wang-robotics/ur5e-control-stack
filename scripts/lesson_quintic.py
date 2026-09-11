"""教学实验: 亲手"看见"五阶多项式 (quintic) 轨迹。

轨迹规划的核心数学就是这个 s(t)。运行后会在 docs\ 下生成
quintic_曲线.png, 三个子图分别是:
    位置 s(t)      —— S 形曲线: 起停都是"平着进、平着出"
    速度 s'(t)     —— 从 0 平滑升到峰值再回到 0
    加速度 s''(t)  —— 从 0 出发、连续变化、回到 0 (没有跳变=没有冲击)

用法:
    python scripts/lesson_quintic.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_ctrl.trajectory import quintic_scale

T = 2.0  # 时长 2 秒
ts = np.linspace(0.0, T, 400)
s = np.array([quintic_scale(t, T)[0] for t in ts])
ds = np.array([quintic_scale(t, T)[1] for t in ts])
dds = np.array([quintic_scale(t, T)[2] for t in ts])

print("关键数值:")
print(f"  t=0  : s={s[0]:.4f}  速度={ds[0]:.4f}  加速度={dds[0]:.4f}")
print(f"  t=T/2: s={s[200]:.4f}  速度={ds[200]:.4f}  加速度={dds[200]:.4f}")
print(f"  t=T  : s={s[-1]:.4f}  速度={ds[-1]:.4f}  加速度={dds[-1]:.4f}")
print(f"  峰值速度 = {ds.max():.4f} (出现在 {ts[np.argmax(ds)]:.2f}s)")

import matplotlib

matplotlib.use("Agg")  # 无窗口, 直接存图
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
axes[0].plot(ts, s, "b-", linewidth=2)
axes[0].set_ylabel("位置 s(t)")
axes[0].set_title("quintic 五阶多项式轨迹 (T=2s)")
axes[0].axhline(0, color="k", linewidth=0.5)
axes[0].axhline(1, color="k", linewidth=0.5)
axes[1].plot(ts, ds, "g-", linewidth=2)
axes[1].set_ylabel("速度 s'(t)")
axes[1].axhline(0, color="k", linewidth=0.5)
axes[2].plot(ts, dds, "r-", linewidth=2)
axes[2].set_ylabel("加速度 s''(t)")
axes[2].set_xlabel("时间 t (s)")
axes[2].axhline(0, color="k", linewidth=0.5)

out = Path(__file__).resolve().parent.parent / "docs" / "quintic_曲线.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(str(out), dpi=120)
print(f"\n图片已保存: {out}")
