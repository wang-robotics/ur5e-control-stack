"""收掉卡点42悬案: FLICK 扫掠, 找"最小脱困甩力"。"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "lesson_singularity.py"

rows = ["flick(Nm)   final_dist_cm   verdict"]
for f in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "flick", "--flick", str(f),
         "--duration", "8", "--headless"],
        capture_output=True, text=True, encoding="gbk", errors="ignore",
    )
    m = re.search(r"最终离目标\s*([\d.]+)m", r.stdout)
    dist = float(m.group(1)) if m else float("nan")
    verdict = "ESCAPED" if dist < 0.05 else "STUCK"
    rows.append(f"{f:<10} {dist * 100:>12.1f}   {verdict}")

out = Path(__file__).resolve().parent / "flick_sweep_result.txt"
out.write_text("\n".join(rows), encoding="utf-8")
print("saved:", out)
