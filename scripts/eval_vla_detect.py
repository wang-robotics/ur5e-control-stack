"""VLA v4.5 评估: 完整可部署系统 —— RGB-D 感知定位 + 目标条件策略。

与 v4 的区别: 目标坐标不再来自仿真真值, 而是从相机图像实时检测:
    1. RGB 找红像素质心/最近点
    2. 深度图取该处距离 (+球半径 = 球心距离)
    3. 反投影到世界坐标 (估计值, 带 ~4cm 感知误差)
    4. 估计目标喂给策略网络

成功标准与之前完全一致: 手离"真值球心" 9cm 内停 1 秒。
成功率相比 v4(真值) 掉多少 = 感知误差的真实代价。

用法: python scripts/eval_vla_detect.py [--episodes 10] [--gain 1.5] [--no-ema]
"""
import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer  # noqa: F401
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
H = W = 224
SUCCESS_DIST = 0.09
SUCCESS_TICKS = 50
MAX_STEPS = 500
BALL_RADIUS = 0.06


class RGBDBallDetector:
    """RGB-D 红球检测器: 渲染图 -> 估计的球心世界坐标。

    2026-09-06 修复: 旧版取"红像素中深度最小的点" (min), 结果被球边缘
    1~2 个异常深度像素劫持 (RGB 与深度通道在轮廓处不一致), 误差 30cm+。
    新版: 红像素质心 + 中位深度 + 球半径, 误差 ~5cm。
    """

    def __init__(self, renderer: mujoco.Renderer) -> None:
        self.renderer = renderer

    def detect(self, data: mujoco.MjData) -> np.ndarray | None:
        # 1. RGB: 找红像素
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data)
        rgb = self.renderer.render().astype(int)
        red = (
            (rgb[:, :, 0] > 120)
            & (rgb[:, :, 0] - rgb[:, :, 1] > 40)
            & (rgb[:, :, 0] - rgb[:, :, 2] > 40)
        )
        ys, xs = np.where(red)
        if len(xs) < 5:
            return None
        # 2. 深度: 红像素的**中位**深度 (抗边缘异常值) + 球半径 = 球心深度
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data)
        depth = self.renderer.render()
        u = float(xs.mean())   # 质心 (对半影月牙有偏, 但稳定)
        v = float(ys.mean())
        d = float(np.median(depth[red])) + BALL_RADIUS
        # 3. 反投影
        sc = self.renderer.scene.camera[0]
        pos_cam = np.array(sc.pos)
        fwd = np.array(sc.forward)
        up = np.array(sc.up)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        up2 = np.cross(right, fwd)
        up2 /= np.linalg.norm(up2)
        f_px = (H / 2) / (abs(sc.frustum_top) / sc.frustum_near)
        x = (u - H / 2) * d / f_px
        y = -(v - W / 2) * d / f_px
        return pos_cam + x * right + y * up2 + d * fwd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/vla_bc.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gain", type=float, default=1.5)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--success", type=float, default=SUCCESS_DIST,
                        help="成功距离阈值(米), 默认 0.09; 与 PPO 对比时用 0.02")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallVLA().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    robot = Robot(xml_path=str(SCENE))
    iface = MuJoCoInterface(robot=robot)
    iface.connect()
    renderer = mujoco.Renderer(robot.model, W, H)
    detector = RGBDBallDetector(renderer)
    rng = np.random.default_rng(42)

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(robot.model, robot.data)

    mean = IMAGENET_MEAN.reshape(1, 1, 3)
    std = IMAGENET_STD.reshape(1, 1, 3)

    successes = 0
    for ep in range(args.episodes):
        iface.reset()
        tcp0, _ = iface.get_tcp_pose()
        goal = tcp0 + np.array(
            [rng.uniform(-0.05, 0.05), rng.uniform(-0.22, -0.12), rng.uniform(-0.06, 0.06)]
        )
        goal[2] = max(goal[2], 0.15)
        robot.model.body("target_ball").pos = goal
        mujoco.mj_forward(robot.model, robot.data)

        near_ticks = 0
        success = False
        min_dist = float("inf")
        ema_action = None
        goal_est = None
        blind_steps = 0  # 检测失败的步数
        for _ in range(MAX_STEPS):
            # 感知: 从图像检测目标
            est = detector.detect(robot.data)
            if est is not None:
                goal_est = est if goal_est is None else 0.5 * goal_est + 0.5 * est
            else:
                blind_steps += 1
            # 回退策略 (真实部署没有真值可用):
            #   有历史估计 -> 沿用; 从未成功检测 -> 用当前手位置 (中性目标, 不虚构)
            if goal_est is not None:
                g_in = goal_est
            else:
                hand_now, _ = iface.get_tcp_pose()
                g_in = hand_now
            # 策略: 图像 + 估计目标 -> 动作
            renderer.disable_depth_rendering()
            renderer.update_scene(robot.data)
            img = renderer.render()
            x = ((img.astype(np.float32) / 255.0) - mean) / std
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
            g_t = torch.from_numpy(np.asarray(g_in, dtype=np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(x, g_t)[0].cpu().numpy()
            action = action * args.gain
            if args.no_ema:
                ema_action = action
            elif ema_action is None:
                ema_action = action
            else:
                ema_action = 0.7 * ema_action + 0.3 * action
            iface.set_tcp_velocity(ema_action)
            iface.step()

            hand, _ = iface.get_tcp_pose()
            dist = float(np.linalg.norm(goal - hand))
            min_dist = min(min_dist, dist)
            if dist < args.success:
                near_ticks += 1
                if near_ticks >= SUCCESS_TICKS:
                    success = True
                    break
            else:
                near_ticks = 0
            if viewer is not None:
                viewer.sync()

        hand, _ = iface.get_tcp_pose()
        dist = float(np.linalg.norm(goal - hand))
        successes += int(success)
        blind = "检测全程失明!" if goal_est is None else f"失明{blind_steps}步"
        tag = "成功" if success else "失败"
        print(f"episode {ep + 1:2d}: {tag}  最终距离 {dist * 100:5.1f} cm  "
              f"最近距离 {min_dist * 100:5.1f} cm  ({blind})")

    print(f"\n成功率: {successes}/{args.episodes}")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
