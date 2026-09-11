"""数据可信度补课: PPO 统一 50 局评估, 输出 均值±标准差。

用法:
    python scripts/eval_ppo.py                      # 评估默认 3 个种子模型
    python scripts/eval_ppo.py --models outputs/ur5e_reach_ppo.zip
    python scripts/eval_ppo.py --episodes 50

输出 (每模型):
    成功率 x/50, 最终距离 均值±标准差, 最近距离 均值±标准差
最后汇总: 跨种子 成功率均值±标准差 (成败按伯努利分布统计)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 允许从任意目录运行: 项目根加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO

from arm_ctrl.envs import UR5eReachEnv

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
SUCCESS_DIST = 0.02  # PPO 口径: 2cm


def evaluate_one(model_path: Path, episodes: int, seed_base: int) -> dict:
    env = UR5eReachEnv()
    model = PPO.load(str(model_path))
    successes = 0
    finals: list[float] = []
    mins: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        min_d = float("inf")
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            tcp, _ = env.interface.get_tcp_pose()
            d = float(np.linalg.norm(env.goal_pos - tcp))
            min_d = min(min_d, d)
            done = terminated or truncated
        finals.append(d)
        mins.append(min_d)
        successes += int(d < SUCCESS_DIST)
    env.close()
    return {
        "successes": successes,
        "episodes": episodes,
        "final_mean": float(np.mean(finals)),
        "final_std": float(np.std(finals)),
        "min_mean": float(np.mean(mins)),
        "min_std": float(np.std(mins)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=[
            str(OUTPUT_DIR / "ur5e_reach_ppo_s0.zip"),
            str(OUTPUT_DIR / "ur5e_reach_ppo_s1.zip"),
            str(OUTPUT_DIR / "ur5e_reach_ppo_s2.zip"),
        ],
    )
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    per_seed_sr: list[float] = []
    print(f"评估 {len(args.models)} 个模型, 每模型 {args.episodes} 局, 成功阈值 2cm", flush=True)
    for i, mp in enumerate(args.models):
        p = Path(mp)
        if not p.exists():
            print(f"跳过 (不存在): {p}", flush=True)
            continue
        r = evaluate_one(p, args.episodes, seed_base=1000 * i)
        sr = r["successes"] / r["episodes"]
        per_seed_sr.append(sr)
        print(
            f"{p.name}: 成功率 {r['successes']}/{r['episodes']} ({sr*100:.1f}%) | "
            f"最终距离 {r['final_mean']*100:.2f}±{r['final_std']*100:.2f} cm | "
            f"最近距离 {r['min_mean']*100:.2f}±{r['min_std']*100:.2f} cm",
            flush=True,
        )

    if len(per_seed_sr) > 1:
        arr = np.array(per_seed_sr)
        print(
            f"\n跨种子汇总: 成功率 {arr.mean()*100:.1f}% ± {arr.std()*100:.1f}% "
            f"(n={len(arr)} 个种子 × {args.episodes} 局)",
            flush=True,
        )


if __name__ == "__main__":
    main()
