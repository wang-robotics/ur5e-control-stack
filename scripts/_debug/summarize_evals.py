"""汇总 outputs/ 下各次 VLA/PPO 评估的逐局距离, 输出均值±标准差。

用法: python scripts/_debug/summarize_evals.py
"""
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

FILES = [
    "outputs/eval_vla_2cm_mindist.txt",   # 2cm 口径, 有最近距离
    "outputs/eval_vla_9cm.txt",           # 9cm 口径 (20局)
    "outputs/eval_vla_v3_50ep.txt",       # v3 复现, 9cm 口径 50局
    "outputs/eval_vla_v4_50ep.txt",       # v4 真值目标, 9cm 口径 50局
    "outputs/eval_vla_v45_50ep.txt",      # v4.5 检测目标(旧版), 9cm 口径 50局
    "outputs/eval_vla_v4_2cm_50ep.txt",   # v4 真值目标, 2cm 口径 50局
    "outputs/eval_vla_v45_fixed_9cm_50ep.txt",  # v4.5 检测目标(修复版), 9cm 口径 50局
    "outputs/eval_vla_v45_fixed_2cm_50ep.txt",  # v4.5 检测目标(修复版), 2cm 口径 50局
]


def read_any(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ("utf-8", "utf-16", "gbk"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="ignore")


pat = re.compile(r"episode\s+\d+:\s+(\S+)\s+最终距离\s+([\d.]+)\s+cm(?:\s+最近距离\s+([\d.]+)\s+cm)?")
pat_sr = re.compile(r"成功率:\s*(\d+)/(\d+)")

for rel in FILES:
    p = ROOT / rel
    if not p.exists():
        print(f"{rel}: 文件不存在")
        continue
    txt = read_any(p)
    finals, mins = [], []
    for m in pat.finditer(txt):
        finals.append(float(m.group(2)))
        if m.group(3):
            mins.append(float(m.group(3)))
    sr = pat_sr.search(txt)
    sr_s = f"{sr.group(1)}/{sr.group(2)}" if sr else "?"
    f = np.array(finals)
    line = f"{rel}: 成功率 {sr_s} | 最终距离 {f.mean():.2f}±{f.std():.2f} cm (n={len(f)})"
    if mins:
        mn = np.array(mins)
        line += f" | 最近距离 {mn.mean():.2f}±{mn.std():.2f} cm (n={len(mn)})"
    print(line)
