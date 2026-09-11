"""校准 + 验证 RGB-D 检测器全流程。

流程 (与最终检测器完全相同):
    1. 渲染 RGB -> 找红像素质心 (u, v)
    2. 渲染深度图 -> 取质心处深度值
    3. 深度归一化值 -> 真实距离 (线性拟合)
    4. 反投影 -> 3D 世界坐标
    5. 与球的真值位置对比, 输出误差
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

SCENE = (
    Path(__file__).resolve().parent.parent.parent
    / "models" / "ur5e" / "universal_robots_ur5e" / "teleop_scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_ball")
renderer = mujoco.Renderer(model, 224, 224)

H = W = 224
FOVY = 45.0
F = (H / 2) / np.tan(np.deg2rad(FOVY / 2))
CX = CY = H / 2

cam_pos = model.cam_pos[0].copy()
cam_mat = model.cam_mat0[0].reshape(3, 3).copy()


def detect_goal() -> tuple:
    """RGB-D 检测: 返回 (估计的球心世界坐标, 红像素数)。"""
    renderer.disable_depth_rendering()
    renderer.update_scene(data)
    rgb = renderer.render().astype(int)
    red = (rgb[:, :, 0] > 120) & (rgb[:, :, 0] - rgb[:, :, 1] > 40) & (rgb[:, :, 0] - rgb[:, :, 2] > 40)
    ys, xs = np.where(red)
    if len(xs) == 0:
        return None, 0
    u, v = float(xs.mean()), float(ys.mean())
    renderer.enable_depth_rendering()
    renderer.update_scene(data)
    depth = renderer.render()
    # 方案1: 质心像素的深度 (旧)
    # 方案2: 红像素中深度最小的点 (球面离相机最近, 恰在相机-球心射线上)
    dmin = float(depth[red].min())
    idx = np.argmin(depth[red])
    u2, v2 = float(xs[idx]), float(ys[idx])
    return (u, v, float(depth[int(v), int(u)]), u2, v2, dmin), len(xs)


# 两个已知位置校准
positions = [[-0.12, 0.33, 0.50], [-0.10, 0.27, 0.42], [-0.05, 0.30, 0.55]]
samples = []
for pos in positions:
    model.body_pos[bid] = pos
    mujoco.mj_forward(model, data)
    det, nred = detect_goal()
    if det is None:
        print(f"球 {pos}: 未检测到红像素")
        continue
    u, v, dval, u2, v2, dmin = det
    p_cam = cam_mat.T @ (np.array(pos) - cam_pos)
    d_true = -p_cam[2]
    samples.append((u, v, dval, d_true, pos, nred, u2, v2, dmin))
    print(f"球 {pos}: 质心({u:.0f},{v:.0f}) 深度 {dval:.3f} | 最近点({u2:.0f},{v2:.0f}) 深度 {dmin:.3f} | 红px {nred}")

if len(samples) >= 2:
    vals = np.array([s[2] for s in samples])   # dval
    trues = np.array([s[3] for s in samples])  # d_true
    A = np.vstack([vals, np.ones_like(vals)]).T
    coef, *_ = np.linalg.lstsq(A, trues, rcond=None)
    print(f"\n深度映射拟合: 距离 = {coef[0]:.4f} * 深度值 + {coef[1]:.4f}")

    # 反投影对比: 方案1(质心) vs 方案2(最近点+半径)
    print("\n反投影测试 (误差单位 cm):")
    errs1, errs2 = [], []
    for u, v, dval, _dt, pos, _nred, u2, v2, dmin in samples:
        sc = renderer.scene.camera[0]
        pos_cam = np.array(sc.pos)
        fwd = np.array(sc.forward)
        up = np.array(sc.up)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        up2 = np.cross(right, fwd)
        up2 /= np.linalg.norm(up2)
        tan_half = abs(sc.frustum_top) / sc.frustum_near
        f_px = (H / 2) / tan_half

        def backproj(uu, vv, dd):
            x = (uu - CX) * dd / f_px
            y = -(vv - CY) * dd / f_px
            return pos_cam + x * right + y * up2 + dd * fwd

        est1 = backproj(u, v, dval + 0.06)     # 质心 + 半径
        est2 = backproj(u2, v2, dmin + 0.06)   # 最近点 + 半径
        e1 = np.linalg.norm(est1 - np.array(pos)) * 100
        e2 = np.linalg.norm(est2 - np.array(pos)) * 100
        errs1.append(e1)
        errs2.append(e2)
        print(f"  球 {pos}: 质心法 {e1:5.1f} cm | 最近点法 {e2:5.1f} cm")
    print(f"\n质心法平均 {np.mean(errs1):.1f} cm | 最近点法平均 {np.mean(errs2):.1f} cm")
