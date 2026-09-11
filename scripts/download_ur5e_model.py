"""一次性脚本: 从 GitHub 下载 mujoco_menagerie 的 UR5e 模型到本地 models/。

之后项目不再依赖 robot_descriptions 的 git 克隆机制。
"""
import json
import urllib.request
from pathlib import Path

API = "https://api.github.com/repos/google-deepmind/mujoco_menagerie/contents"
RAW = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main"
ROOT = Path(__file__).resolve().parent.parent / "models" / "ur5e"


def fetch_json(url: str) -> list:
    req = urllib.request.Request(url, headers={"User-Agent": "dsh"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def download(rel_path: str) -> None:
    dest = ROOT / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"skip {rel_path}")
        return
    url = f"{RAW}/{rel_path}"
    req = urllib.request.Request(url, headers={"User-Agent": "dsh"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"downloaded {rel_path} ({dest.stat().st_size} bytes)")


def walk(rel_dir: str) -> None:
    for item in fetch_json(f"{API}/{rel_dir}"):
        rel = f"{rel_dir}/{item['name']}"
        if item["type"] == "dir":
            walk(rel)
        elif item["type"] == "file":
            download(rel)


if __name__ == "__main__":
    walk("universal_robots_ur5e")
    print("done ->", ROOT)
