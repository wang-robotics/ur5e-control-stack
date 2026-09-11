"""VLA 模型评估: 在仿真里闭环跑 10 局, 统计成功率。

用法:
    python scripts/eval_vla.py                 # 用默认模型 outputs/vla_bc.pt
    python scripts/eval_vla.py --headless      # 不弹窗
    python scripts/eval_vla.py --episodes 20

判定: 手进入球 5cm 内停留 1 秒 = 成功 (与遥操作同样的标准)。
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer  # noqa: F401  viewer 需显式导入
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
SUCCESS_DIST = 0.09  # 与遥操作教材一致: 离球心 9cm 内停 1 秒算成功
SUCCESS_TICKS = 50   # 1 秒
MAX_STEPS = 500      # 每局最多 10 秒


def main() -> None:
    global SUCCESS_DIST
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/vla_bc.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gain", type=float, default=1.0,
                        help="动作增益: 模型输出偏小(回归到均值)时乘这个系数")
    parser.add_argument("--no-ema", action="store_true", help="关闭动作平滑")
    parser.add_argument("--success", type=float, default=SUCCESS_DIST,
                        help="成功距离阈值(米), 默认 0.09; 与 PPO 对比时用 0.02")
    parser.add_argument("--no-goal", action="store_true",
                        help="纯图像版 (v3 复现): 模型不看目标坐标")
    args = parser.parse_args()
    SUCCESS_DIST = args.success

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallVLA(goal_cond=not args.no_goal).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"模型已加载: {args.model}")

    robot = Robot(xml_path=str(SCENE))
    iface = MuJoCoInterface(robot=robot)
    iface.connect()
    renderer = mujoco.Renderer(robot.model, 224, 224)
    rng = np.random.default_rng(42)

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(robot.model, robot.data)

    mean = IMAGENET_MEAN.reshape(1, 1, 3)
    std = IMAGENET_STD.reshape(1, 1, 3)

    successes = 0
    for ep in range(args.episodes):
        # 随机开局: 复位 + 随机放球 (与采集同分布)
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
        ema_action = None  # 动作平滑 (指数移动平均)
        for _ in range(MAX_STEPS):
            renderer.update_scene(robot.data)
            img = renderer.render()
            x = ((img.astype(np.float32) / 255.0) - mean) / std
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                if args.no_goal:
                    action = model(x)[0].cpu().numpy()
                else:
                    g_t = torch.from_numpy(goal.astype(np.float32)).unsqueeze(0).to(device)
                    action = model(x, g_t)[0].cpu().numpy()
            action = action * args.gain  # 油门增益
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
            if dist < SUCCESS_DIST:
                near_ticks += 1
                if near_ticks >= SUCCESS_TICKS:
                    success = True
                    break
            else:
                near_ticks = 0
            if viewer is not None:
                viewer.sync()
                time.sleep(0.004)

        hand, _ = iface.get_tcp_pose()
        dist = float(np.linalg.norm(goal - hand))
        successes += int(success)
        print(f"episode {ep + 1:2d}: {'成功' if success else '失败'}  "
              f"最终距离 {dist * 100:5.1f} cm  最近距离 {min_dist * 100:5.1f} cm")

    print(f"\n成功率: {successes}/{args.episodes}")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
