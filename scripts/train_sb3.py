"""Stable-Baselines3 PPO 训练 UR5eReachEnv (末端空间动作)。

用法:
    python scripts/train_sb3.py                       # 默认 20 万步
    python scripts/train_sb3.py --timesteps 1000000   # 更长训练
    python scripts/train_sb3.py --eval                # 只加载模型评估

训练产物:
    outputs/ur5e_reach_ppo.zip   训练好的模型
    outputs/tb_logs/             TensorBoard 日志
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

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


def train(timesteps: int) -> PPO:
    vec_env = DummyVecEnv([make_env()])
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=str(OUTPUT_DIR / "tb_logs"),
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=timesteps)
    model_path = OUTPUT_DIR / "ur5e_reach_ppo.zip"
    model.save(str(model_path))
    print(f"模型已保存: {model_path}")
    vec_env.close()
    return model


def evaluate(model_path: str, episodes: int = 10) -> None:
    env = UR5eReachEnv()
    model = PPO.load(model_path)
    successes = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        # 判定: 以最终距离为准
        tcp, _ = env.interface.get_tcp_pose()
        d = float(np.linalg.norm(env.goal_pos - tcp))
        successes += int(d < 0.02)
        print(f"episode {ep + 1}: reward={total_reward:.1f}, dist={d:.4f}")
    print(f"成功率: {successes}/{episodes}")
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="训练/评估 UR5eReach")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument(
        "--eval",
        action="store_true",
        help="只评估已训练模型 (outputs/ur5e_reach_ppo.zip)",
    )
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    if args.eval:
        model_path = args.model or str(OUTPUT_DIR / "ur5e_reach_ppo.zip")
        evaluate(model_path)
    else:
        train(args.timesteps)


if __name__ == "__main__":
    main()
