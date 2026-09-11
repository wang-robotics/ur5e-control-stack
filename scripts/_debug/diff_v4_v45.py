"""核对 v4 (真值目标) 与 v4.5 (检测目标) 50 局评估的逐局距离是否完全一致。

若逐局完全相同 → v4.5 的检测可能一直返回 None, 静默回退到真值目标
(eval_vla_detect.py 第 129 行: g_in = goal if goal_est is None else goal_est),
即 v4.5 的 50/50 实际上又测了一遍 v4。
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def read(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ("utf-8", "utf-16", "gbk"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="ignore")


pat = re.compile(r"episode\s+(\d+):\s+\S+\s+最终距离\s+([\d.]+)")

a = {int(m.group(1)): float(m.group(2)) for m in pat.finditer(read(ROOT / "outputs/eval_vla_v4_50ep.txt"))}
b = {int(m.group(1)): float(m.group(2)) for m in pat.finditer(read(ROOT / "outputs/eval_vla_v45_50ep.txt"))}

same = [k for k in sorted(a) if k in b and abs(a[k] - b[k]) < 1e-9]
diff = [k for k in sorted(a) if k in b and abs(a[k] - b[k]) >= 1e-9]
print(f"v4 局数: {len(a)}, v4.5 局数: {len(b)}")
print(f"逐局距离完全相同的局数: {len(same)} / {min(len(a), len(b))}")
print(f"不同的局: {diff}")
for k in diff[:10]:
    print(f"  ep{k}: v4={a[k]:.3f}cm  v4.5={b[k]:.3f}cm  差={abs(a[k]-b[k])*100:.1f}cm")
