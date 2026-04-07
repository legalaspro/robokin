# Placo Examples

Examples using the [Placo](https://github.com/Rhoban/placo) IK backend with [Viser](https://viser.studio/) for 3D visualization.

<video src="https://github.com/user-attachments/assets/f3958cb9-21a6-460e-9a66-509d36a6c9c3" width="600" controls></video>

<table>
<tr>
<td><video src="https://github.com/user-attachments/assets/ca9ddc47-adbd-4dd0-8c6e-e57ba177ee6b" width="400" controls></video></td>
<td><video src="https://github.com/user-attachments/assets/d5ac6d6c-06f5-4219-b0f7-5234ae83877b" width="400" controls></video></td>
</tr>
<tr>
<td><video src="https://github.com/user-attachments/assets/a0870cf8-3b89-4389-8cd6-e3e98c1f7457" width="400" controls></video></td>
<td></td>
</tr>
</table>

## Files

| File | Description |
|------|-------------|
| `interactive_ik.py` | Drag a 3D gizmo to control the end-effector via differential IK. Named-pose buttons generate Cartesian segments to target poses. No hardware required. |
| `go_to_pose.py` | Move the robot to a target pose via two strategies: **joint interpolation** (offline IK, then interpolate joints) and **Cartesian servo** (online IK every tick). |
| `pose_cycle.py` | Cycle through predefined poses using both joint-interp and Cartesian-servo modes. Each loop button toggles start/stop. |
| `rerun_loop.py` | Headless Cartesian servo loop with Rerun logging. Cycles through poses at 50 Hz, streaming joint angles and EE position to Rerun time-series panels. |
| `lerobot_real_arm.py` | Full real-arm demo driving an SO-101 via LeRobot at 50 Hz. Supports manual joint override, interactive IK gizmo, and auto pose loops. Requires a connected SO-101 arm and LeRobot (see below). |
| `ros2_real_arm.py` | Placeholder for ROS 2 real-arm integration. Requires `so101-ros-physical-ai` and a ROS 2 workspace. |

## Quick Start

```bash
# Install robokin with placo + viser extras
pip install -e ".[placo,viser]"

# Run an example (no hardware needed)
python examples/placo/interactive_ik.py
```

Open the Viser viewer at [http://localhost:8080](http://localhost:8080).

### LeRobot (for real-arm examples)

The `lerobot_real_arm.py` example requires [LeRobot](https://github.com/huggingface/lerobot) with the Feetech motor driver:

```bash
pip install lerobot "lerobot[feetech]"
```
