"""演示视频渲染 v2 —— 按评估协议渲染 "漂亮且诚实" 的片段。

相对 v1 (render_demo_frames.py) 的四个改进:
  1. **用 teleop 固定机位** (v1 用默认自由相机, 视角差);
  2. **控制协议与评估一致**: 增益 1.5、**无 EMA** 平滑 (v1 用了 EMA, 与报告口径不符);
  3. **达成即止**: 末端进入成功阈值并保持 1 秒 (= 官方成功判据) 后停止录制,
     再静止 hold 若干帧收尾 —— 避免 v1 "跑满 400 步导致手臂飘离球体";
  4. **PPO 也在带球的场景里渲染** (teleop_scene.xml), 观众能看到目标球;
     并记录每局最终距离, 失败局直接丢弃 (视频只展示成功片段)。

用法:
    python scripts/render_demo_v2.py --mode vla --episodes 3
    python scripts/render_demo_v2.py --mode ppo --episodes 3
    python scripts/render_demo_v2.py --mode both --episodes 3

输出: outputs/videos2/{vla,ppo}_v2_ep*/  (PNG 帧, 用 make_video.py 合成 mp4)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from arm_ctrl import config  # noqa: E402
from arm_ctrl.interfaces import MuJoCoInterface  # noqa: E402
from arm_ctrl.robot import Robot  # noqa: E402
from train_vla import IMAGENET_MEAN, IMAGENET_STD, SmallVLA  # noqa: E402

SCENE = ROOT / "models" / "ur5e" / "universal_robots_ur5e" / "teleop_scene.xml"
OUTDIR = ROOT / "outputs" / "videos2"
CAMERA = "teleop"          # (仅作物位参考: 实测该机位看不到工作区, 已改用下面的自由相机)
CAM_DISTANCE = 1.35        # 自由相机: 距工作区中点
CAM_AZIMUTH = 210.0        # 水平角 (实测 210° 构图最好: 机械臂完整 + 红球清晰)
CAM_ELEVATION = -20.0      # 俯角
GAIN = 1.5                 # 与 eval_vla_detect.py 默认一致
CTRL_HZ = 50               # 控制频率
HOLD_FRAMES = 12           # 成功后静止收尾的帧数


def _make_camera(lookat) -> mujoco.MjvCamera:
    """按工作区中点生成自由相机 (每局对准 起点与目标的中间)。"""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = CAM_DISTANCE
    cam.azimuth = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION
    return cam


def _render(renderer: mujoco.Renderer, data: mujoco.MjData, camera=None) -> np.ndarray:
    """camera=None 用模型默认相机 (**必须**用于喂策略: 训练/评估都是默认相机);
    传 MjvCamera 只用于出片视角。"""
    if camera is None:
        renderer.update_scene(data)
    else:
        renderer.update_scene(data, camera=camera)
    return renderer.render()


def _save(img: np.ndarray, outdir: Path, idx: int) -> None:
    from PIL import Image

    Image.fromarray(img).save(outdir / f"{idx:04d}.png")


def _policy_vla(device: str):
    net = SmallVLA().to(device)
    net.load_state_dict(torch.load(ROOT / "outputs" / "vla_bc.pt", map_location=device))
    net.eval()
    return net


def render_vla(episodes: int, fps: int, success: float, keep_failures: bool) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = _policy_vla(device)
    robot = Robot(xml_path=str(SCENE))
    iface = MuJoCoInterface(robot=robot)
    iface.connect()

    model_renderer = mujoco.Renderer(robot.model, 224, 224)   # 喂策略
    frame_renderer = mujoco.Renderer(robot.model, 480, 480)   # 出片
    mean = IMAGENET_MEAN.reshape(1, 1, 3)
    std = IMAGENET_STD.reshape(1, 1, 3)
    step_every = max(1, CTRL_HZ // fps)
    rng = np.random.default_rng(42)          # 与 eval_vla_detect.py 相同的种子
    max_steps = 500           # 10 s 上限 (与评估一致)
    success_ticks = CTRL_HZ   # 1 秒

    kept = 0
    ep = 0
    while kept < episodes and ep < episodes * 6:
        ep += 1
        iface.reset()
        tcp0, _ = iface.get_tcp_pose()
        # 目标采样: **与 eval_vla_detect.py L120-124 完全一致** (口径对齐)
        goal = tcp0 + np.array([
            rng.uniform(-0.05, 0.05),
            rng.uniform(-0.22, -0.12),
            rng.uniform(-0.06, 0.06),
        ])
        goal[2] = max(goal[2], 0.15)
        robot.model.body("target_ball").pos = goal
        mujoco.mj_forward(robot.model, robot.data)

        cam = _make_camera((tcp0 + goal) / 2)      # 每局对准 起点↔目标 中点
        outdir = OUTDIR / f"vla_v2_ep{kept}"
        outdir.mkdir(parents=True, exist_ok=True)
        for f in outdir.glob("*.png"):
            f.unlink()

        frame = 0
        in_success = 0
        final = float("nan")
        for step in range(max_steps):
            img = _render(model_renderer, robot.data)
            x = ((img.astype(np.float32) / 255.0) - mean) / std
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
            g = torch.from_numpy(goal.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                a = net(x, g)[0].cpu().numpy() * GAIN       # 增益 1.5, 不做 EMA
            iface.set_tcp_velocity(a)
            iface.step()

            tcp, _ = iface.get_tcp_pose()
            final = float(np.linalg.norm(goal - tcp))
            in_success = in_success + 1 if final < success else 0

            if step % step_every == 0:
                _save(_render(frame_renderer, robot.data, cam), outdir, frame)
                frame += 1
            if in_success >= success_ticks:                  # 达标满 1 秒 → 收尾
                break

        for _ in range(HOLD_FRAMES):                         # 静止收尾
            _save(_render(frame_renderer, robot.data, cam), outdir, frame)
            frame += 1

        ok = final < success
        print(f"VLA 尝试{ep}: {frame} 帧  最终距离 {final*100:5.2f} cm  "
              f"{'✅' if ok else '❌ 丢弃'}  -> {outdir.relative_to(ROOT)}")
        if ok:
            kept += 1
        elif not keep_failures:
            for f in outdir.glob("*.png"):
                f.unlink()
            outdir.rmdir()
    iface.disconnect()


def render_ppo(episodes: int, fps: int, success: float, keep_failures: bool) -> None:
    from stable_baselines3 import PPO

    model = PPO.load(ROOT / "outputs" / "ur5e_reach_ppo.zip")
    robot = Robot(xml_path=str(SCENE))
    iface = MuJoCoInterface(robot=robot)
    iface.connect()
    frame_renderer = mujoco.Renderer(robot.model, 480, 480)
    step_every = max(1, CTRL_HZ // fps)
    rng = np.random.default_rng(5)
    r_min, r_max = config.REACH_GOAL_OFFSET
    max_steps, success_ticks = 200, CTRL_HZ // 2   # 2cm 达标即可 (无需停 1 秒)

    kept, ep = 0, 0
    while kept < episodes and ep < episodes * 6:
        ep += 1
        iface.reset()
        tcp0, _ = iface.get_tcp_pose()
        for _ in range(100):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            goal = tcp0 + d * rng.uniform(r_min, r_max)
            if goal[2] >= config.REACH_GOAL_Z_MIN:
                break
        robot.model.body("target_ball").pos = goal
        mujoco.mj_forward(robot.model, robot.data)

        cam = _make_camera((tcp0 + goal) / 2)      # 每局对准 起点↔目标 中点
        outdir = OUTDIR / f"ppo_v2_ep{kept}"
        outdir.mkdir(parents=True, exist_ok=True)
        for f in outdir.glob("*.png"):
            f.unlink()

        frame, in_success, final = 0, 0, float("nan")
        for step in range(max_steps):
            tcp, _ = iface.get_tcp_pose()
            rel = goal - tcp
            obs = np.concatenate([
                iface.get_joint_positions(), iface.get_joint_velocities(), tcp, goal, rel,
            ]).astype(np.float32)
            action, _ = model.predict(obs, deterministic=True)
            iface.set_tcp_velocity(np.asarray(action, dtype=float) * config.REACH_ACTION_SCALE)
            iface.step()

            tcp, _ = iface.get_tcp_pose()
            final = float(np.linalg.norm(goal - tcp))
            in_success = in_success + 1 if final < success else 0
            if step % step_every == 0:
                _save(_render(frame_renderer, robot.data, cam), outdir, frame)
                frame += 1
            if in_success >= success_ticks:
                break

        for _ in range(HOLD_FRAMES):
            _save(_render(frame_renderer, robot.data, cam), outdir, frame)
            frame += 1

        ok = final < success
        print(f"PPO 尝试{ep}: {frame} 帧  最终距离 {final*100:5.2f} cm  "
              f"{'✅' if ok else '❌ 丢弃'}  -> {outdir.relative_to(ROOT)}")
        if ok:
            kept += 1
        elif not keep_failures:
            for f in outdir.glob("*.png"):
                f.unlink()
            outdir.rmdir()
    iface.disconnect()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vla", "ppo", "both"], default="both")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--vla-success", type=float, default=0.09)
    ap.add_argument("--ppo-success", type=float, default=0.02)
    ap.add_argument("--keep-failures", action="store_true")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.mode in ("vla", "both"):
        print("=== VLA (行为克隆, 9cm 口径) ===")
        render_vla(args.episodes, args.fps, args.vla_success, args.keep_failures)
    if args.mode in ("ppo", "both"):
        print("=== PPO (强化学习, 2cm 口径) ===")
        render_ppo(args.episodes, args.fps, args.ppo_success, args.keep_failures)
    print("\n完成。合成 mp4:")
    print("  python scripts/make_video.py --indir outputs/videos2/<dir> "
          "--out outputs/videos2/<dir>.mp4 --fps 25")


if __name__ == "__main__":
    main()
