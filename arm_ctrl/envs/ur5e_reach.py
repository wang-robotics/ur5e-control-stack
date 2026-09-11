"""UR5e Reach: 末端到达空间随机目标点的 Gymnasium 环境。

- action (6,): 归一化末端速度 [-1, 1] x [v_lim, w_lim] -> OSC 流控
- obs (21,):  qpos(6) + qvel(6) + tcp_pos(3) + goal_pos(3) + goal_rel(3)
- reward:     接近塑形 + 指数接近奖励 + 动作/速度惩罚 + 成功奖励
- 成功:       tcp 距目标 < 2cm; 单 episode 100 步 (2s)

环境只依赖 RobotInterface, 与 MuJoCo 解耦 (换真机实现可复用)。
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

from .. import config
from ..interfaces import MuJoCoInterface, RobotInterface


class UR5eReachEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        interface: Optional[RobotInterface] = None,
        render_mode: Optional[str] = None,
        horizon: int = config.REACH_EPISODE_STEPS,
        success_dist: float = config.REACH_SUCCESS_DIST,
        goal_offset: tuple[float, float] = config.REACH_GOAL_OFFSET,
    ) -> None:
        super().__init__()
        self.interface = interface if interface is not None else MuJoCoInterface()
        self.render_mode = render_mode
        self.horizon = horizon
        self.success_dist = success_dist
        self.goal_offset = goal_offset

        nq = config.HOME_QP0.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(nq + nq + 3 + 3 + 3,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self.goal_pos = np.zeros(3)
        self._step_count = 0
        self._prev_dist = 0.0
        self._viewer = None
        self._renderer = None

    # ------------------------------------------------------------------ 采样
    def _sample_goal(self, rng: np.random.Generator) -> np.ndarray:
        """以当前 TCP 为中心采样目标 (偏移 8~25cm, 桌面以上, 保证可达)。

        Reach 任务标准做法: 目标相对初始末端位置采样, 避免初始距离
        超出 episode 时长内的最大行程。
        """
        tcp0, _ = self.interface.get_tcp_pose()
        r_min, r_max = self.goal_offset
        for _ in range(100):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            g = tcp0 + direction * rng.uniform(r_min, r_max)
            if g[2] >= config.REACH_GOAL_Z_MIN:
                return g
        return tcp0 + np.array([0.12, 0.0, 0.0])

    def _get_obs(self) -> np.ndarray:
        q = self.interface.get_joint_positions()
        qd = self.interface.get_joint_velocities()
        tcp, _ = self.interface.get_tcp_pose()
        rel = self.goal_pos - tcp
        return np.concatenate(
            [q, qd, tcp, self.goal_pos, rel]
        ).astype(np.float32)

    # ------------------------------------------------------------- gym API
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        rng = self.np_random

        # 初始位姿: home 附近小扰动
        q0 = config.HOME_QP0 + rng.uniform(-0.08, 0.08, size=6)
        self.interface.reset(q0)
        self.goal_pos = self._sample_goal(rng)

        tcp, _ = self.interface.get_tcp_pose()
        self._prev_dist = float(np.linalg.norm(self.goal_pos - tcp))
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 归一化动作 -> 末端速度 -> OSC 流控
        twist = np.asarray(action, dtype=float) * config.REACH_ACTION_SCALE
        self.interface.set_tcp_velocity(twist)
        self.interface.step()  # 一个控制周期 (含物理子步)
        self._step_count += 1

        tcp, _ = self.interface.get_tcp_pose()
        dist = float(np.linalg.norm(self.goal_pos - tcp))
        qvel = self.interface.get_joint_velocities()

        # 奖励塑形: 接近塑形 + 近距离指数奖励 + 动作/速度惩罚
        reward = -2.0 * (dist - self._prev_dist)
        reward += 0.2 * np.exp(-40.0 * dist)
        reward -= 0.005 * float(np.linalg.norm(twist) ** 2)
        reward -= 0.0005 * float(np.linalg.norm(qvel) ** 2)
        self._prev_dist = dist

        success = dist < self.success_dist
        terminated = bool(success) or tcp[2] < 0.02
        truncated = self._step_count >= self.horizon
        if success:
            reward += 20.0

        info = {"distance": dist, "success": success}
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ 渲染
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            self._render_human()
            return None
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        raise ValueError(f"未知 render_mode: {self.render_mode}")

    def _render_human(self) -> None:
        import mujoco
        import mujoco.viewer  # noqa: F401  显式导入 viewer 子模块

        if self._viewer is None:
            model, data = self.interface.model, self.interface.data
            try:
                self._viewer = mujoco.viewer.launch_passive(model, data)
            except AttributeError:  # 极老/新版本兼容
                self._viewer = mujoco.viewer.launch(model, data)
        self._viewer.sync()

    def _render_rgb(self) -> np.ndarray:
        import mujoco

        model, data = self.interface.model, self.interface.data
        if self._renderer is None:
            self._renderer = mujoco.Renderer(model, 480, 480)
        self._renderer.update_scene(data)
        return self._renderer.render()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self.interface.disconnect()
