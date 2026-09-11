"""把渲染好的 PNG 帧序列合成为 mp4 (需要 imageio-ffmpeg)。

用法:
    python scripts/make_video.py --indir outputs/videos/vla_ep0 --out outputs/videos/vla_ep0.mp4 --fps 15
    python scripts/make_video.py --all   # 合成所有帧序列
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VIDEODIR = ROOT / "outputs" / "videos"


def make_video(indir: Path, out: Path, fps: int) -> None:
    import imageio.v2 as imageio

    frames = sorted(indir.glob("*.png"))
    if not frames:
        print(f"{indir}: 没有帧")
        return
    writer = imageio.get_writer(out, fps=fps)
    for f in frames:
        writer.append_data(imageio.imread(f))
    writer.close()
    print(f"已合成: {out} ({len(frames)} 帧 @ {fps}fps)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        for indir in sorted(VIDEODIR.glob("*")):
            if indir.is_dir():
                make_video(indir, indir.with_suffix(".mp4"), args.fps)
    else:
        if not args.indir:
            print("需要 --indir 或 --all")
            return
        indir = Path(args.indir)
        out = Path(args.out) if args.out else indir.with_suffix(".mp4")
        make_video(indir, out, args.fps)


if __name__ == "__main__":
    main()
