"""教学实验: 奇异位形脱困 —— DLS 卡死 vs 惯性甩出 (W2 Day 2 加餐二)。

场景: 两连杆臂 (各长 1m) 完全对折 (奇异位形), 手就在基座处。
任务: 让手沿"对折方向"伸出去 0.3m —— 这正是奇异位形丢失的方向。

两种模式:
    dls   : 纯 DLS 求解 (阻尼最小二乘), 预期: 卡死 (手几乎不动)
    flick : 先给关节2 一个扭矩脉冲"甩一下", 再上 DLS, 预期: 脱困

你控制甩的力度 (两种方式任选):
    方式1: 改文件顶部的 FLICK / DURATION / SLOW 三个大写常量, 保存后直接运行
    方式2: 命令行覆盖: python scripts/lesson_singularity.py --flick 3 --slow 4
    --headless  = 不弹窗

输出: 每 0.25s 打印 手的位置 / 离目标距离 / 可操作性指标 (雅可比行列式)。
可操作性 = 0 表示正处于奇异位形; 越大越"健康"。
"""
import argparse
import sys
import time

import mujoco
import mujoco.viewer  # noqa: F401  viewer 需显式导入
import numpy as np

L1 = L2 = 1.0
Q_FOLD = np.array([np.deg2rad(45.0), np.deg2rad(180.0)])  # 对折奇异位形
DT_CTRL = 0.02          # 控制周期 20ms
LAMBDA = 0.05           # DLS 阻尼
FLICK = 0.01             # ★ 甩的力度(牛米): 改这个数! 0=不甩, 1~5 自己试
DURATION = 16.0         # ★ 实验总时长(秒)
SLOW = 2.0              # ★ 慢放倍数: 1=实时, 2=慢一倍, 4=慢四倍

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <light dir="0 0 -1" pos="0 0 3"/>
    <geom type="plane" size="3 3 0.1" rgba=".9 .9 .9 1"/>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 1 0 0" size="0.03" mass="1" rgba=".2 .4 .8 1"/>
      <body name="link2" pos="1 0 0">
        <joint name="j2" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 1 0 0" size="0.03" mass="1" rgba=".8 .4 .2 1"/>
        <site name="hand" pos="1 0 0" size="0.03"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="a1" joint="j1" gear="1" ctrlrange="-100 100"/>
    <motor name="a2" joint="j2" gear="1" ctrlrange="-100 100"/>
  </actuator>
</mujoco>
"""


def jacobian(q: np.ndarray) -> np.ndarray:
    """两连杆平面臂的解析雅可比 (2x2)。"""
    t1, t2 = q
    s1, s12 = np.sin(t1), np.sin(t1 + t2)
    c1, c12 = np.cos(t1), np.cos(t1 + t2)
    J = np.array(
        [
            [-L1 * s1 - L2 * s12, -L2 * s12],
            [L1 * c1 + L2 * c12, L2 * c12],
        ]
    )
    return J


def manipulability(q: np.ndarray) -> float:
    return float(abs(np.linalg.det(jacobian(q))))


def pd_torque(model, data, q_des, kp=200.0, kd=10.0) -> np.ndarray:
    return kp * (q_des - data.qpos[:2]) - kd * data.qvel[:2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dls", "flick"], default="flick")
    parser.add_argument("--flick", type=float, default=None,
                        help="甩的力度(牛米), 不写则用文件顶部的 FLICK")
    parser.add_argument("--duration", type=float, default=None,
                        help="总时长(秒), 不写则用文件顶部的 DURATION")
    parser.add_argument("--slow", type=float, default=None,
                        help="慢放倍数, 不写则用文件顶部的 SLOW")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--lambda", dest="lam", type=float, default=None,
                        help="DLS 阻尼 λ, 不写则用文件顶部的 LAMBDA")
    args = parser.parse_args()

    # 命令行没给参数时, 使用文件顶部的大写常量 (方便在代码里直接改)
    if args.flick is None:
        args.flick = FLICK
    if args.duration is None:
        args.duration = DURATION
    if args.slow is None:
        args.slow = SLOW
    if args.lam is not None:
        global LAMBDA
        LAMBDA = args.lam

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[:2] = Q_FOLD
    mujoco.mj_forward(model, data)

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(model, data)

    # 目标: 沿"对折方向"伸出去 0.3m (奇异位形丢失的方向)
    link_dir = np.array([np.cos(Q_FOLD[0] + np.pi), np.sin(Q_FOLD[0] + np.pi)])
    target = 0.3 * link_dir

    q_des = Q_FOLD.copy()
    t = 0.0
    print(f"模式: {args.mode}, 甩的力度: {args.flick} Nm")
    print(f"目标: 手要伸到 {np.round(target, 3)}")
    print("-" * 64)

    while t < args.duration:
        # 阶段1 (0~0.5s): 稳定在对折位形
        if t < 0.5:
            data.ctrl[:] = pd_torque(model, data, Q_FOLD)
        else:
            # 阶段2: flick 模式先甩 0.15s
            if args.mode == "flick" and t < 0.65:
                tau = pd_torque(model, data, Q_FOLD)
                tau[1] += args.flick  # 给关节2 一个扭矩脉冲
                data.ctrl[:] = tau
            else:
                # DLS: 把目标方向速度映射成关节速度, 积分成位置目标
                hand = data.site("hand").xpos[:2]
                v = np.clip(3.0 * (target - hand), -0.5, 0.5)
                J = jacobian(data.qpos[:2])
                JJt = J @ J.T
                dq = J.T @ np.linalg.solve(JJt + LAMBDA**2 * np.eye(2), v)
                q_des += dq * DT_CTRL
                data.ctrl[:] = pd_torque(model, data, q_des)

        mujoco.mj_step(model, data, 10)
        t += DT_CTRL

        if int(round(t / DT_CTRL)) % 12 == 0:  # 每 0.25s 打印一次
            hand = data.site("hand").xpos[:2]
            dist = float(np.linalg.norm(target - hand))
            print(
                f"t={t:4.2f}s  手=({hand[0]:+.3f},{hand[1]:+.3f})  "
                f"离目标={dist:.4f}m  可操作性={manipulability(data.qpos[:2]):.4f}"
            )
        if viewer is not None:
            viewer.sync()
            time.sleep(DT_CTRL * args.slow)  # 慢放倍数越大, 看得越清楚

    hand = data.site("hand").xpos[:2]
    dist = float(np.linalg.norm(target - hand))
    print("-" * 64)
    verdict = "脱困成功!" if dist < 0.05 else "卡死了 (手没出来)"
    print(f"最终离目标 {dist:.4f}m -> {verdict}")


if __name__ == "__main__":
    main()
