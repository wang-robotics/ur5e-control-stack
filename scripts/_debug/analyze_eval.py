"""分析 VLA 评估日志: 按最终距离分类成功/失败 (中文被重定向破坏, 用距离判定)。"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r"outputs\eval_vla_20.txt"
raw = open(path, "rb").read()
text = None
for enc in ("utf-16", "utf-8", "gbk"):
    try:
        text = raw.decode(enc)
        break
    except Exception:
        continue

succs, fails = [], []
for m in re.finditer(r"episode\s+(\d+):", text):
    tail = text[m.end(): m.end() + 40]
    dm = re.search(r"([\d.]+)\s*cm", tail)
    if not dm:
        continue
    dist = float(dm.group(1))
    if dist < 9.5:
        succs.append((int(m.group(1)), dist))
    else:
        fails.append((int(m.group(1)), dist))

total = len(succs) + len(fails)
print(f"总 {total} 局 | 成功 {len(succs)} ({100*len(succs)/total:.0f}%) | 失败 {len(fails)}")
if succs:
    print("成功局距离:", [round(d, 1) for _, d in succs])
if fails:
    print("失败局距离:", [round(d, 1) for _, d in fails])
