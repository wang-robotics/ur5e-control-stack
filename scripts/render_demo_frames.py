"""演示视频预录: 渲染 VLA 与 PPO 的闭环动画帧 (PNG 序列 -> 后续合成为 mp4)。

两阶段:
    阶段1 (本脚本): 渲染帧到 outputs/videos/frames_*/
    阶段2: ffmpeg 可用后合成 mp4 (或直接把 PNG 序列给剪辑软件)

用法:
    python scripts/render_demo_frames.py --mode vla --episodes 2 --fps 15
    python scripts/render_demo_frames.py --mode ppo --episodes 2 --fps 15
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_ctrl.interfaces import MuJoCoInterface
from arm_ctrl.robot import Robot
from train_vla import IMAGENET_MEAN, IMAGENET_STD, SmallVLA

SCENE = (
    Path(__file__).resolve().parent.parent
    / "models" / "ur5e" / "universal_robots_ur5e" / "teleop_scene.xml"
)
OUTDIR = Path(__file__).resolve().parent.parent / "outputs" / "videos"


def render_vla(episodes: int, fps: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_net = SmallVLA().to(device)
    model_net.load_state_dict(
        torch.load("outputs/vla_bc.pt", map_location=device)
    )
    model_net.eval()
    robot = Robot(xml_path=str(SCENE))
    iface = MuJoCoInterface(robot=robot)
    iface.connect()
    model_renderer = mujoco.Renderer(robot.model, 224, 224)   # 喂模型用
    frame_renderer = mujoco.Renderer(robot.model, 480, 480)   # 视频帧用 (480上限)
    rng = np.random.default_rng(7)
    mean = IMAGENET_MEAN.reshape(1, 1, 3)
    std = IMAGENET_STD.reshape(1, 1, 3)
    step_every = max(1, 50 // fps)  # 控制 50Hz, 渲染 fps

    for ep in range(episodes):
        iface.reset()
        tcp0, _ = iface.get_tcp_pose()
        goal = tcp0 + np.array(
            [rng.uniform(-0.05, 0.05), rng.uniform(-0.22, -0.12), rng.uniform(-0.06, 0.06)]
        )
        goal[2] = max(goal[2], 0.15)
        robot.model.body("target_ball").pos = goal
        mujoco.mj_forward(robot.model, robot.data)
        outdir = OUTDIR / f"vla_ep{ep}"
        outdir.mkdir(parents=True, exist_ok=True)
        ema = None
        frame = 0
        for step in range(400):  # 8 秒
            model_renderer.update_scene(robot.data)
            img = model_renderer.render()
            x = ((img.astype(np.float32) / 255.0) - mean) / std
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
            g = torch.from_numpy(goal.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                a = model_net(x, g)[0].cpu().numpy() * 1.5
            ema = a if ema is None else 0.7 * ema + 0.3 * a
            iface.set_tcp_velocity(ema)
            iface.step()
            if step % step_every == 0:
                from PIL import Image

                frame_renderer.update_scene(robot.data)
                big = frame_renderer.render()
                Image.fromarray(big).save(outdir / f"{frame:04d}.png")
                frame += 1
        print(f"VLA episode {ep}: {frame} 帧 -> {outdir}")


def render_ppo(episodes: int, fps: int) -> None:
    # PPO 用 UR5eReachEnv 的策略: 加载 SB3 模型走环境
    from stable_baselines3 import PPO

    from arm_ctrl.envs import UR5eReachEnv

    model_ppo = PPO.load("outputs/ur5e_reach_ppo.zip")
    env = UR5eReachEnv(render_mode="rgb_array")
    step_every = max(1, 50 // fps)
    for ep in range(episodes):
        obs, _ = env.reset()
        outdir = OUTDIR / f"ppo_ep{ep}"
        outdir.mkdir(parents=True, exist_ok=True)
        frame = 0
        for step in range(100):  # 2 秒
            action, _ = model_ppo.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if step % step_every == 0:
                img = env.render()
                from PIL import Image

                Image.fromarray(img).save(outdir / f"{frame:04d}.png")
                frame += 1
            if terminated or truncated:
                break
        print(f"PPO episode {ep}: {frame} 帧 -> {outdir}")
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vla", "ppo"], default="vla")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    if args.mode == "vla":
        render_vla(args.episodes, args.fps)
    else:
        render_ppo(args.episodes, args.fps)
    print("帧渲染完成 (合成 mp4 需 ffmpeg, 或直接把 PNG 序列导入剪辑软件)")


if __name__ == "__main__":
    main()
