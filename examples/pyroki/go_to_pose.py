# Copyright 2026 Dmitri Manajev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PyRoki + Viser "go to pose" example using robokin.

No live gizmo tracking and no manual joint editing — just buttons
that move the robot to the current target pose using two strategies:

  • Joint interp:    solve IK once, then interpolate joints directly
  • Cartesian servo: interpolate EE pose and solve IK every tick

Usage:
    python examples/pyroki_viser_go_to_pose.py
"""

from __future__ import annotations

import threading
import time

import numpy as np
import viser
import yourdfpy

from robokin.pyroki import PyrokiKinematics, PyrokiConfig
from robokin.robot_model import load_robot_description
from robokin.transformations import (
    compute_segment_steps_from_speed,
    ease_in_out_sine,
    interpolate_pose,
)
from robokin.ui.viser_app import ViserRobotUI


# ── Robot config ──────────────────────────────────────────────────────

EE_LINK = "gripper_frame_link"
DT = 1.0 / 50.0

def main() -> None:
    # ── Load robot model ──
    model = load_robot_description("so_arm101_description")
    urdf = yourdfpy.URDF.load(str(model.urdf_path))

    # ── Create solver ──
    solver = PyrokiKinematics(
        urdf=urdf,
        ee_link_name=EE_LINK,
        cfg=PyrokiConfig(dt=DT, mode="basic_ik_collision"),
    )

    # ── Warm up JAX (first solve triggers JIT compilation) ──
    print("Warming up JAX…")
    solver.warmup()
    print("JAX ready.")

    # ── Initial state ──
    q_init = solver.get_joint_state()
    T_init = solver.current_pose()

    # ── Viser UI (robot viz + EE readouts only) ──
    server = viser.ViserServer()
    ui = ViserRobotUI(server=server, urdf=urdf,
                      gripper_joint_name="gripper")
    ui.build(initial_q=q_init, initial_T=T_init, enable_gripper=True)


    # Gripper slider → manual FK
    @ui.gripper_slider.on_update
    def _(event) -> None:
        q = ui.get_joint_values()
        solver.set_joint_state(q)
        T = solver.current_pose()
        ui.update_robot_from_joint_values(q)
        ui.update_ee_display(T)
        ui.set_target_pose(T)

    # ── Pose helpers ──────────────────────────────────────────────────

    def _go_to_joint_target(T_target: np.ndarray) -> None:
        """
        Move to ``T_target`` by solving IK once, then interpolating in joint space.

        The target pose is converted to a joint-space goal with a single IK solve.
        Motion then proceeds by interpolating joint values directly from the current
        configuration to that goal.

        The final end-effector pose matches the IK solution, but the Cartesian path
        during motion is not explicitly constrained.
        """
        def _run():
            with ui.busy():
                q_start = solver.get_joint_state()
                T_start = solver.current_pose()  # capture before solve_goal clobbers state
                q_target = solver.solve_goal(q_start, T_target)

                n_steps = compute_segment_steps_from_speed(
                    T_start=T_start,
                    T_goal=T_target,
                    dt=DT,
                    linear_speed_mps=solver.cfg.linear_speed_mps,
                    angular_speed_radps=solver.cfg.angular_speed_radps,
                )

                for k in range(1, n_steps + 1):
                    alpha = ease_in_out_sine(k / n_steps)
                    q = q_start + alpha * (q_target - q_start)
                    solver.set_joint_state(q)
                    ui.sync_from_solver(solver)
                    time.sleep(DT)
        threading.Thread(target=_run, daemon=True).start()

    def _go_to_pose_cartesian_servo(T_target: np.ndarray) -> None:
        """
        Move to ``T_target`` by interpolating end-effector poses and solving IK
        at each control step.

        The robot tracks a Cartesian reference trajectory from the current pose
        to ``T_target``. This better preserves the intended tool motion than
        joint-space interpolation.
        """
        def _run():
            with ui.busy():
                T_start = solver.current_pose()
                n_steps = compute_segment_steps_from_speed(
                    T_start=T_start,
                    T_goal=T_target,
                    dt=DT,
                    linear_speed_mps=solver.cfg.linear_speed_mps,
                    angular_speed_radps=solver.cfg.angular_speed_radps,
                )
                for k in range(1, n_steps + 1):
                    alpha = ease_in_out_sine(k / n_steps)
                    T_ref = interpolate_pose(T_start, T_target, alpha)
                    q = solver.solve_goal(solver.get_joint_state(), T_ref)
                    solver.set_joint_state(q)
                    ui.sync_from_solver(solver)
                    time.sleep(DT)
        threading.Thread(target=_run, daemon=True).start()

    # ── Buttons ───────────────────────────────────────────────────────

    go_to_pose_joint_btn = server.gui.add_button("Go to target (joint interp)")

    @go_to_pose_joint_btn.on_click
    def _(event) -> None:
        T_target = ui.get_target_pose()
        _go_to_joint_target(T_target)

    go_to_pose_servo_btn = server.gui.add_button("Go to target (pose servo)")

    @go_to_pose_servo_btn.on_click
    def _(event) -> None:
        T_target = ui.get_target_pose()
        _go_to_pose_cartesian_servo(T_target)


    # ── Keep alive ────────────────────────────────────────────────────
    print("\n═══════════════════════════════════════════════════════")
    print("  robokin PyRoki + Viser (go-to-pose) — http://localhost:8080")
    print("  • Press buttons to animate to named poses")
    print("  • Ctrl+C to exit")
    print("═══════════════════════════════════════════════════════\n")

    server.sleep_forever()


if __name__ == "__main__":
    main()

