"""机器人模型封装: 加载 / 正运动学 FK / 逆运动学 IK / 雅可比。

- FK: 通过 MuJoCo 的 mj_kinematics 直接读取 site 位姿 (零开销)
- IK: 自研阻尼最小二乘 (DLS) 迭代, 对奇异位形鲁棒, 零外部依赖
- Jacobian: 中心差分数值雅可比 (对 6 轴臂足够快, 且无版本兼容风险)

坐标约定: 位置为世界系 (m), 姿态四元数为 wxyz。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import config


class Robot:
    """UR5e (或任意 MuJoCo 铰链臂) 的运动学封装。"""

    def __init__(
        self,
        xml_path: Optional[str | Path] = None,
        tcp_site: str = config.TASK_SITE,
        home_qpos: Optional[np.ndarray] = None,
    ) -> None:
        """加载模型。

        Args:
            xml_path: MJCF 路径; 为 None 时加载项目内置的 Menagerie UR5e 模型。
            tcp_site: 末端 TCP 参考 site 名。
            home_qpos: 初始关节角, 默认 config.HOME_QP0。
        """
        if xml_path is None:
            # 优先用项目内自带的 Menagerie UR5e 模型 (models/ur5e)
            local_scene = (
                Path(__file__).resolve().parent.parent
                / "models"
                / "ur5e"
                / "universal_robots_ur5e"
                / "scene.xml"
            )
            if local_scene.exists():
                xml_path = local_scene
            else:
                # fallback: robot_descriptions (需要 git 克隆)
                from robot_descriptions.ur5e_mj_description import UR5E_MJCF

                xml_path = UR5E_MJCF

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self._tcp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, tcp_site
        )
        if self._tcp_site_id < 0:
            names = [self.model.site(i).name for i in range(self.model.nsite)]
            raise ValueError(f"site '{tcp_site}' 不存在, 可用 site: {names}")

        self.tcp_site = tcp_site

        # 默认仅控制前 nv 个自由度 (UR5e: 6)
        self.nq = self.model.nq
        self.nv = self.model.nv
        if home_qpos is None:
            home_qpos = config.HOME_QP0
        self.home_qpos = np.asarray(home_qpos, dtype=float)
        if self.home_qpos.shape != (self.nq,):
            raise ValueError(
                f"home_qpos 长度 {self.home_qpos.shape} 与模型 nq={self.nq} 不符"
            )

        self.reset()

    # ------------------------------------------------------------------ 状态
    def reset(self, qpos: Optional[np.ndarray] = None) -> None:
        """重置仿真到给定(或 home)位姿, 速度为 0。"""
        q = self.home_qpos if qpos is None else np.asarray(qpos, dtype=float)
        self.data.qpos[: self.nq] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    @property
    def qpos(self) -> np.ndarray:
        return self.data.qpos[: self.nq].copy()

    @property
    def qvel(self) -> np.ndarray:
        return self.data.qvel[: self.nv].copy()

    def _update_kinematics(self) -> None:
        """只更新运动学 (不积分动力学), 用于 FK/雅可比。"""
        mujoco.mj_kinematics(self.model, self.data)

    # -------------------------------------------------------------------- FK
    def fk(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """正运动学: 关节角 -> TCP 位姿。

        Returns:
            (pos(3,), quat(4,) wxyz)
        """
        q = np.asarray(qpos, dtype=float)
        if q.shape != (self.nq,):
            raise ValueError(f"qpos 长度应为 {self.nq}, 收到 {q.shape}")
        self.data.qpos[: self.nq] = q
        self._update_kinematics()
        xmat = self.data.site_xmat[self._tcp_site_id].reshape(3, 3)
        quat = np.roll(Rotation.from_matrix(xmat).as_quat(), 1)  # xyzw -> wxyz
        return self.data.site_xpos[self._tcp_site_id].copy(), quat

    def _fk_matrix(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """FK 返回 (pos, 3x3 旋转矩阵), 供雅可比等内部计算使用。"""
        q = np.asarray(qpos, dtype=float)
        self.data.qpos[: self.nq] = q
        self._update_kinematics()
        return (
            self.data.site_xpos[self._tcp_site_id].copy(),
            self.data.site_xmat[self._tcp_site_id].reshape(3, 3).copy(),
        )

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """读取当前仿真状态的 TCP 位姿 (pos, quat wxyz)。"""
        pos = self.data.site_xpos[self._tcp_site_id].copy()
        rot = Rotation.from_matrix(
            self.data.site_xmat[self._tcp_site_id].reshape(3, 3)
        ).as_quat()  # xyzw
        quat = np.roll(rot, 1)  # -> wxyz
        return pos, quat

    # -------------------------------------------------------------------- IK
    def ik(
        self,
        pos: np.ndarray,
        quat: Optional[np.ndarray] = None,
        q0: Optional[np.ndarray] = None,
        max_iters: int = 200,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
        damping: float = 2e-2,
    ) -> tuple[np.ndarray, bool]:
        """逆运动学: TCP 目标位姿 -> 关节角 (阻尼最小二乘迭代)。

        基于数值雅可比的 DLS 迭代 (Levenberg-Marquardt 风格):
            dq = J^T (J J^T + lambda^2 I)^{-1} err
        配合关节步长裁剪与限位, 对奇异位形鲁棒、零外部依赖。

        Args:
            pos: 目标位置 (3,)
            quat: 目标姿态 wxyz (4,), None 表示只约束位置
            q0: 迭代初值, 默认当前位姿
            max_iters: 最大迭代次数
            pos_tol: 位置收敛容差 (m)
            rot_tol: 姿态收敛容差 (rad)
            damping: DLS 阻尼系数

        Returns:
            (q(6,), success)
        """
        q = self.data.qpos[: self.nq].copy() if q0 is None else q0.copy()
        pos_target = np.asarray(pos, dtype=float)
        R_target = None
        if quat is not None:
            R_target = Rotation.from_quat(
                np.roll(np.asarray(quat), -1)
            ).as_matrix()

        success = False
        for _ in range(max_iters):
            pos_cur, R_cur = self._fk_matrix(q)
            err_p = pos_target - pos_cur
            rotvec = None
            if R_target is not None:
                rotvec = Rotation.from_matrix(
                    R_target @ R_cur.T
                ).as_rotvec()
                err = np.concatenate([err_p, rotvec])
            else:
                err = err_p

            err_pos = float(np.linalg.norm(err_p))
            err_rot = (
                float(np.linalg.norm(rotvec)) if rotvec is not None else 0.0
            )
            if err_pos < pos_tol and err_rot < rot_tol:
                success = True
                break

            jac = self.jacobian(q)
            if R_target is None:
                jac = jac[:3]

            # 阻尼最小二乘
            jjt = jac @ jac.T
            reg = np.eye(jjt.shape[0]) * damping**2
            dq = jac.T @ np.linalg.solve(jjt + reg, err)
            # 步长裁剪 (防跳跃, 增强稳定)
            dq = np.clip(dq, -0.2, 0.2)
            q_new = np.clip(
                q + dq,
                config.JOINT_LIMITS[:, 0],
                config.JOINT_LIMITS[:, 1],
            )
            if float(np.max(np.abs(q_new - q))) < 1e-7:
                break  # 停滞: 局部极小或限位卡死
            q = q_new
        return q, success

    # --------------------------------------------------------------- Jacobian
    def jacobian(self, qpos: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """数值雅可比 J(q) in R^{6 x nv}: 前 3 行线速度, 后 3 行角速度。

        基于中心差分, 对 6 轴臂一次计算 ~13 次 mj_kinematics, 开销可忽略。
        """
        q = np.asarray(qpos, dtype=float)
        jac = np.zeros((6, self.nv))
        for i in range(self.nv):
            qp, qm = q.copy(), q.copy()
            qp[i] += eps
            qm[i] -= eps
            pp, Rp = self._fk_matrix(qp)
            pm, Rm = self._fk_matrix(qm)
            jac[:3, i] = (pp - pm) / (2.0 * eps)
            # 角速度: log(R(q+eps) * R(q-eps)^T) / (2 eps)
            dR = Rp @ Rm.T
            jac[3:, i] = Rotation.from_matrix(dR).as_rotvec() / (2.0 * eps)
        return jac

    def osc_joint_velocities(
        self,
        qpos: np.ndarray,
        twist: np.ndarray,
        damping: float = config.OSC_DLS_LAMBDA,
    ) -> np.ndarray:
        """OSC: 末端 6 维速度 (线速度 + 角速度) -> 关节速度 (阻尼最小二乘)。

        Args:
            qpos: 当前关节角
            twist: [vx, vy, vz, wx, wy, wz]
            damping: DLS 正则系数
        """
        jac = self.jacobian(qpos)
        # J^+ = J^T (J J^T + lambda^2 I)^{-1}
        jjt = jac @ jac.T
        reg = damping**2 * np.eye(6)
        dq = jac.T @ np.linalg.solve(jjt + reg, np.asarray(twist, dtype=float))
        return np.clip(dq, -config.JOINT_VEL_LIMIT, config.JOINT_VEL_LIMIT)
