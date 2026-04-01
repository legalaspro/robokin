# PyRoki Examples

Examples using the [PyRoki](https://github.com/chungmin99/pyroki) IK backend with [Viser](https://viser.studio/) for 3D visualization.

## Files

| File | Description |
|------|-------------|
| `interactive_ik.py` | Drag a 3D gizmo to control the end-effector in real time. Demonstrates `ViserRobotUI` + `PyrokiKinematics` with callback-driven IK — no hardware required. |
| `go_to_pose.py` | Move the robot to a target pose via two strategies: **joint interpolation** (solve IK once, interpolate joints) and **Cartesian servo** (interpolate EE pose, solve IK every tick). |
| `pose_cycle.py` | Cycle through a list of predefined poses using both joint-interp and Cartesian-servo modes. Each loop button toggles start/stop. |
| `lerobot_real_arm.py` | Full real-arm demo driving an SO-101 via LeRobot at 50 Hz. Supports manual joint override, interactive IK gizmo, and auto pose loops. Requires a connected SO-101 arm and LeRobot (see below). |

## Quick Start

```bash
# Install robokin with pyroki + viser extras
pip install -e ".[pyroki,viser]"

# Run an example (no hardware needed)
python examples/pyroki/interactive_ik.py
```

Open the Viser viewer at [http://localhost:8080](http://localhost:8080).

### LeRobot (for real-arm examples)

The `lerobot_real_arm.py` example requires [LeRobot](https://github.com/huggingface/lerobot) with the Feetech motor driver:

```bash
pip install lerobot "lerobot[feetech]"
```
