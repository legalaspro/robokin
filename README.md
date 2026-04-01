# 🦾 Robokin

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Alpha-orange)]()

**URDF-based forward and inverse kinematics helpers for robot arms.**

Robokin provides a clean, backend-agnostic API for robot kinematics with pluggable IK solvers, 3D visualization, and real-arm integration.

## ✨ Features

- **Multiple IK backends** — [Placo](https://github.com/Rhoban/placo), [PyRoki](https://github.com/chungmin99/pyroki), RoboPlan
- **URDF-first** — load any robot from [robot_descriptions](https://github.com/robot-descriptions/robot_descriptions.py) or a local URDF file
- **3D visualization** — built-in [Viser](https://viser.studio/) UI with interactive gizmo control
- **Rerun logging** — stream joint angles and EE trajectories to [Rerun](https://rerun.io/) dashboards
- **Real-arm support** — drive hardware via [LeRobot](https://github.com/huggingface/lerobot) or ROS 2

## 🎬 Showcases

<table>
<tr>
<td><strong>Rerun IK Loop (Placo)</strong></td>
<td><strong>Pose Cycle (PyRoki)</strong></td>
</tr>
<tr>
<td><video src="docs/assets/rerun_loop.mp4" width="400" controls></video></td>
<td><video src="docs/assets/pose_cycle.mp4" width="400" controls></video></td>
</tr>
</table>

> More demo videos will be uploaded as new examples and backends are added.

## 📦 Installation

```bash
# Core (numpy, scipy, robot_descriptions)
pip install -e .

# With a specific IK backend
pip install -e ".[placo]"
pip install -e ".[pyroki]"

# With visualization
pip install -e ".[viser]"
pip install -e ".[rerun]"

# Everything
pip install -e ".[all]"
```

### Real-arm (LeRobot + Feetech)

```bash
pip install lerobot "lerobot[feetech]"
```

## 🚀 Quick Start

```python
from robokin import load_robot_description, PyrokiKinematics, PyrokiConfig

# Load SO-101 robot model
model = load_robot_description("so100", ee_link="gripper_frame_link")

# Create kinematics solver
kin = PyrokiKinematics(model.urdf_path, model.joint_names, "gripper_frame_link")

# Solve IK, run servo steps, etc.
```

## 📂 Project Structure

```
robokin/
├── src/robokin/          # Library source
│   ├── robot_model.py    # URDF loading & RobotModel
│   ├── placo.py          # Placo IK backend
│   ├── pyroki.py         # PyRoki IK backend
│   ├── roboplan_oink.py  # RoboPlan backend
│   ├── transformations.py# Pose interpolation & easing
│   ├── pyroki_snippets/  # PyRoki solver variants
│   └── ui/               # Viser & Rerun helpers
├── examples/
│   ├── pyroki/           # PyRoki examples
│   ├── placo/            # Placo examples
│   └── helpers/          # Shared example utilities
├── tests/                # Unit tests
├── pyproject.toml
└── LICENSE
```

## 🧪 Tests

```bash
pip install -e ".[dev]"
pytest
```

## 📄 License

Copyright 2026 Dmitri Manajev. Licensed under the [Apache License 2.0](LICENSE).
