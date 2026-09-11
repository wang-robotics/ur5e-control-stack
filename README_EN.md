# UR5e Robot Arm Control Stack with Reinforcement Learning and VLA

> **Author**: Zhiwei Wang (王智伟) · Guangdong University of Technology,
> Mechanical Design, Manufacturing and Automation (Class of 2022) ·
> lixinnian95@gmail.com

A complete six-axis robot arm control stack built on **MuJoCo**, extended with two
learning-based controllers: **reinforcement learning (PPO)** and **vision-based
behavior cloning (VLA-style)**.

## Results (read the thresholds — they are not comparable)

| Method | Success @ 2 cm | Success @ 9 cm |
|---|---|---|
| PPO (trial and error) | **150/150** (50 episodes × 3 seeds), landing **1.86 ± 0.09 cm** | trivially met |
| Behavior cloning (VLA-style) | **0–10%** (three training runs) | **50/50** |

The cloning ceiling is set by the demonstrations: only about **10%** of my own
teleoperation episodes stop inside 2 cm (mean terminal distance **4.6 ± 2.4 cm**,
measured by physics replay). Substituting ground-truth goals, or a fixed RGB-D
detector, does not change the outcome — the gap is **imitation vs. model**, not
tuning.

## Overview

This project provides a full control stack for a UR5e six-axis robot arm in the
MuJoCo physics simulator, organized into six layers: kinematics, trajectory
planning, control, a communication abstraction layer, a Gymnasium reinforcement
learning environment, and the simulation itself. On top of this stack, two
learning-based controllers were developed and evaluated on the same reach task:
a PPO policy trained purely from rewards, and a lightweight VLA-style policy
trained from keyboard-teleoperation demonstrations with an RGB-D target detector.

## Architecture

| Layer | Module | Responsibility |
|---|---|---|
| Kinematics | `arm_ctrl/robot.py` | Forward/inverse kinematics (damped least squares), numerical Jacobian |
| Trajectory | `arm_ctrl/trajectory.py` | PTP and Cartesian linear trajectories (quintic polynomials) |
| Control | `arm_ctrl/controller.py` | Position, torque PD (+ gravity compensation), and OSC modes |
| Abstraction | `arm_ctrl/interfaces.py` | `RobotInterface`: an 11-method contract so upper layers never touch MuJoCo |
| RL Environment | `arm_ctrl/envs/` | Gymnasium reach environment (21-dim observation, 6-dim end-effector velocity action) |
| Simulation | `models/ur5e/` | Official Menagerie UR5e model (BSD-3, see `THIRD_PARTY.md`) |

**Key design**: the upper layers depend only on the `RobotInterface` abstraction.
Connecting a real robot (e.g., via ROS 2) requires only adding one new
implementation class; the control stack and the policies remain unchanged.

This provides a **command channel** for a future real robot. It is necessary but
**not sufficient** for sim-to-real transfer: the policy itself would still need
domain randomization, system identification, and real-robot fine-tuning.

## Features

- Damped least squares IK, robust to singular configurations (validated by a
  dedicated singularity-escape experiment)
- Quintic-polynomial trajectories with zero start/stop acceleration
- 20 unit tests covering kinematics, trajectories, interfaces, and environments
- RGB-D target detector for the VLA pipeline: red-ball segmentation with
  **centroid + median depth**, **~5 cm** localization error. An earlier version
  used a minimum-depth heuristic that was hijacked by inconsistent edge pixels,
  reaching 20+ cm — the failure and the fix are documented.

## Experiments

### 1. Reinforcement Learning (PPO)

- **Task**: reach a randomly placed target (success = within 2 cm)
- **Training**: 1M simulation steps × 3 seeds (~30 min per seed on CPU),
  Stable-Baselines3 PPO
- **Result**: **150/150 episodes**, landing accuracy **1.86 ± 0.09 cm**
- **Key fix**: sampling goals around the initial end-effector pose keeps them
  inside the 2-second travel budget (0.15 m/s × 2 s = 30 cm). Before the fix,
  early training reached only 2/10.

### 2. VLA-Style Behavior Cloning

- **Data**: 24 keyboard-teleoperation episodes (14 auto-saved with terminal
  distance < 9 cm; 10 manually saved), **4,703 frames** after idle-frame trimming
- **Model**: lightweight CNN; input = 224×224 image + goal coordinates;
  output = 6-dim end-effector velocity
- **Pipeline**: teleoperation → preprocessing → behavior cloning → closed-loop
  deployment through OSC
- **Six versions**: 20% → 100% at the 9 cm threshold (final versions: 50/50)
- **Key diagnoses**: *regression to the mean* (idle-heavy data shrinks the
  output; fixed by 3× loss weighting on moving frames and an output gain of 1.5)
  and *goal conditioning* (a perception module supplies the target coordinate)
- **Honest note**: at the 2 cm threshold the policy scores **0–10%** across three
  training runs — consistent with the ~10% of demonstrations that stop inside
  2 cm. A controlled experiment also showed that *removing* the 10 imperfect
  demonstrations made results worse (0/50 vs 5/50), so no data cleaning was kept.

## Quick Start

```powershell
conda activate mujoco_project
pip install -r requirements.txt

# Watch the PTP / line-trajectory demos (opens a 3D viewer)
python scripts/demo_ptp.py
python scripts/demo_traj.py

# Run unit tests
python -m unittest discover -s tests

# Evaluate the trained PPO policy
python scripts/train_sb3.py --eval

# Evaluate the VLA policy with the RGB-D detector (closed loop)
python scripts/eval_vla_detect.py
```

## Repository Structure

```
arm_ctrl/     core library (6 modules + Gymnasium env)
scripts/      demos, training, evaluation, data pipeline
tests/        unit tests (20 cases)
models/       official UR5e Menagerie model + teleoperation scene (BSD-3)
docs/         experiment report (Markdown + HTML), derivation notes
data/         dataset documentation
```

Datasets (~1 GB) and trained weights are **not committed**: teleoperation data is
reproduced with `scripts/teleop_collect.py` (see `data/README.md`), and model
weights are attached to GitHub Releases.

## Future Work

- Replace the color-threshold detector with a learned detector (e.g., YOLO) for
  arbitrary objects
- Add an LLM task planner to convert natural-language instructions into goals
- Fine-tune a larger pretrained VLA (e.g., OpenVLA) on the same action interface
- Transfer the stack to a real UR5e through a `ROS2Interface` implementation
