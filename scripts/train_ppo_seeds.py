"""数据可信度补课: 用 3 个随机种子训练 PPO, 每个 100 万步。

目的: 回答"10/10 是不是运气好"—— 单次训练的成功率有方差,
      多种子训练 + 统一 50 局评估才能给出 均值±标准差。

用法:
    python scripts/train_ppo_seeds.py --seeds 0 1 2 --timesteps 1000000
产物:
    outputs/ur5e_reach_ppo_s0.zip   (seed=0)
    outputs/ur5e_reach_ppo_s1.zip   (seed=1)
    outputs/ur5e_reach_ppo_s2.zip   (seed=2)
注意: 不覆盖 outputs/ur5e_reach_ppo.zip (原始模型保留)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许从任意目录运行: 项目根加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from arm_ctrl.envs import UR5eReachEnv

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def make_env():
    def _make():
        return UR5eReachEnv()

    return _make


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    args = parser.parse_args()

    for seed in args.seeds:
        print(f"\n===== seed={seed} 开始训练 ({args.timesteps} 步) =====", flush=True)
        vec_env = DummyVecEnv([make_env()])
        model = PPO(
            "MlpPolicy",
            vec_env,
            seed=seed,
            verbose=1,
            tensorboard_log=str(OUTPUT_DIR / "tb_logs"),
            n_steps=2048,
            batch_size=256,
            learning_rate=3e-4,
        )
        model.learn(total_timesteps=args.timesteps)
        model_path = OUTPUT_DIR / f"ur5e_reach_ppo_s{seed}.zip"
        model.save(str(model_path))
        vec_env.close()
        print(f"seed={seed} 完成, 已保存: {model_path}", flush=True)

    print("\n全部种子训练完成。", flush=True)


if __name__ == "__main__":
    main()
