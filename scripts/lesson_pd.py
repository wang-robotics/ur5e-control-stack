"""教学实验: PD 控制 —— Kp/Kd 的角色与振荡 (W3 Day 1)。

场景: 一根摆 (长1m, 重1kg, 挂点可绕 y 轴转), 受重力。
任务: 用 PD 控制把它举到 45° 并停住。
     tau = Kp·(目标角-当前角) - Kd·当前角速度 + 重力补偿

运行 (自己调 Kp/Kd 看现象):
    python scripts/lesson_pd.py --kp 100 --kd 0      # 无阻尼: 永远荡
    python scripts/lesson_pd.py --kp 100 --kd 5      # 欠阻尼: 荡几下才停
    python scripts/lesson_pd.py --kp 100 --kd 12     # 临界阻尼: 最快且不荡
    python scripts/lesson_pd.py --kp 100 --kd 50     # 过阻尼: 慢吞吞爬过去
    --duration 秒数  --headless

输出: 每 0.15s 打印角度/误差; 末尾统计"过冲次数"和"停稳时间"。
"""
import argparse
import sys
import time

import mujoco
import mujoco.viewer  # noqa: F401
import numpy as np

TARGET = np.deg2rad(45.0)  # 目标: 举到 45°
DT_CTRL = 0.01             # 控制周期 10ms

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <light dir="0 0 -1" pos="0 0 3"/>
    <body name="pendulum" pos="0 0 1.5">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.04" mass="1" rgba=".2 .4 .8 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="a" joint="j" gear="1" ctrlrange="-200 200"/>
  </actuator>
</mujoco>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kp", type=float, default=100.0)
    parser.add_argument("--kd", type=float, default=12.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(model, data)

    t = 0.0
    overshoots = 0
    last_err = TARGET - data.qpos[0]
    settle_t = None
    print(f"Kp={args.kp}, Kd={args.kd}, 目标 45°")
    print("-" * 52)

    while t < args.duration:
        q, qd = data.qpos[0], data.qvel[0]
        e = TARGET - q
        # PD + 重力补偿 (qfrc_bias = 重力力矩, 提前抵消)
        tau = args.kp * e - args.kd * qd + data.qfrc_bias[0]
        data.ctrl[0] = tau
        mujoco.mj_step(model, data, 5)
        t += DT_CTRL

        if e * last_err < 0:  # 误差过零 = 过冲一次
            overshoots += 1
        last_err = e
        if settle_t is None and abs(e) < 0.01 and abs(qd) < 0.05:
            settle_t = t

        if int(round(t / DT_CTRL)) % 15 == 0:
            print(f"t={t:4.2}s  角度={np.rad2deg(q):+6.1f}°  误差={np.rad2deg(e):+6.2f}°")
        if viewer is not None:
            viewer.sync()
            time.sleep(0.004)

    print("-" * 52)
    print(f"过冲次数: {overshoots}   停稳时间: {settle_t if settle_t else '>'+str(args.duration)+'s (没停住)'}")


if __name__ == "__main__":
    main()
