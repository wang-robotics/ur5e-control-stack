# Third-Party Notices

## 1. MuJoCo Menagerie — Universal Robots UR5e

- **Location in this repository**: `models/ur5e/`
- **Upstream**: [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- **License**: BSD-3-Clause (see `models/ur5e/LICENSE` or the upstream repository)
- **Copyright**: Google DeepMind and Universal Robots A/S
- **Modifications by this project**: an additional scene file
  (`teleop_scene.xml`) was added for data collection — it introduces a target ball
  (radius 0.06 m) and a teleoperation camera. The original UR5e model files are
  otherwise unchanged.

## 2. MuJoCo

- **Upstream**: [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco)
- **License**: Apache-2.0
- **Usage**: runtime dependency only (installed via pip, not vendored here).

## 3. Python dependencies

Declared in `requirements.txt` (e.g., `stable-baselines3`, `gymnasium`,
`scipy`, `numpy`, `matplotlib`, `imageio`). Each retains its own license.

---

If you redistribute this project, keep this file and the upstream license files
inside `models/ur5e/` intact.
