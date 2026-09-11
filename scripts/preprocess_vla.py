"""VLA 数据预处理: 合并所有遥操作数据 + 裁剪长段发呆帧。

发呆帧 (全零动作) 连续超过 MAX_IDLE 帧 (0.4s) 的部分裁掉,
短停顿保留 (教模型"到点停住")。输出合并数据集。

用法:
    python scripts/preprocess_vla.py            # 合并 datasets/teleop/*.npz
    python scripts/preprocess_vla.py --max-idle 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np

MAX_IDLE = 10  # 连续全零动作帧数上限 (10 帧 = 0.4 秒)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="datasets/teleop")
    parser.add_argument("--max-idle", type=int, default=MAX_IDLE)
    parser.add_argument("--out", default="datasets/vla_dataset.npz")
    parser.add_argument("--only", type=str, default=None,
                        help="只合并指定回合 (逗号分隔, 如 ep1,ep3): 用于干净数据对照实验")
    args = parser.parse_args()

    files = sorted(Path(args.datadir).glob("*.npz"))
    files = [f for f in files if "selftest" not in f.name]
    if args.only:
        allowed = {s.strip() for s in args.only.split(",") if s.strip()}
        files = [f for f in files if f.stem in allowed]
    if not files:
        print("没有找到数据文件")
        sys.exit(1)

    all_imgs, all_acts, all_goals = [], [], []
    for f in files:
        d = np.load(f)
        imgs, acts = d["images"], d["actions"]
        goal = np.asarray(d["goal"], dtype=np.float32)
        zero = np.all(np.abs(acts) < 1e-6, axis=1)
        keep = np.ones(len(acts), dtype=bool)
        run = 0
        for i in range(len(acts)):
            if zero[i]:
                run += 1
            else:
                run = 0
            if run > args.max_idle:
                keep[i] = False
        n_cut = int((~keep).sum())
        print(f"{f.name}: {len(acts)} 帧 -> {int(keep.sum())} 帧 (裁掉 {n_cut} 帧发呆)")
        all_imgs.append(imgs[keep])
        all_acts.append(acts[keep])
        all_goals.append(np.tile(goal, (int(keep.sum()), 1)))  # 每帧带上目标坐标

    images = np.concatenate(all_imgs).astype(np.uint8)
    actions = np.concatenate(all_acts).astype(np.float32)
    goals = np.concatenate(all_goals).astype(np.float32)
    zero_pct = 100.0 * float(np.mean(np.all(np.abs(actions) < 1e-6, axis=1)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, images=images, actions=actions, goals=goals)
    print(f"\n合并完成: {len(images)} 帧 -> {out}")
    print(f"裁剪后发呆帧占比: {zero_pct:.1f}%")
    print(f"动作幅度均值: {np.mean(np.linalg.norm(actions, axis=1)):.4f}")


if __name__ == "__main__":
    main()
