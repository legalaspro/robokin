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
PyRoki + Viser interactive IK example using robokin.

Demonstrates ViserRobotUI + PyrokiKinematics with callback-driven events.
No hardware required.

Usage:
    python examples/pyroki_viser.py
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

    # ── Named poses (built from joint names → order-independent) ──
    Q_HOME = solver.make_configuration({})  # all zeros
    Q_REST = solver.make_configuration({
        "shoulder_pan":  0.0,
        "shoulder_lift": -np.pi / 2,
        "elbow_flex":    np.pi / 2,
        "wrist_flex":    np.deg2rad(42.97),
        "wrist_roll":    0.0,
    })

    # ── Warm up JAX (first solve triggers JIT compilation) ──
    print("Warming up JAX…")
    solver.warmup()
    print("JAX ready.")

    # ── Initial state ──
    q_init = solver.get_joint_state()
    T_init = solver.current_pose()

    # ── Viser UI ──
    server = viser.ViserServer()
    ui = ViserRobotUI(server=server, urdf=urdf,
                      gripper_joint_name="gripper")
    ui.build(initial_q=q_init, initial_T=T_init, enable_joint_sliders=True, enable_gripper=True)

    # ── Callbacks ─────────────────────────────────────────────────────

    # Slider → manual FK
    for slider in ui.joint_sliders.values():
        @slider.on_update
        def _(event) -> None:
            if not ui.is_manual_joint_mode():
                return
            q = ui.get_joint_values()
            solver.set_joint_state(q)
            T = solver.current_pose()
            ui.update_robot_from_joint_values(q)
            ui.update_ee_display(T)
            ui.set_target_pose(T)

    # Gripper slider → manual FK
    @ui.gripper_slider.on_update
    def _(event) -> None:
        q = ui.get_joint_values()
        solver.set_joint_state(q)
        T = solver.current_pose()
        ui.update_robot_from_joint_values(q)
        ui.update_ee_display(T)
        ui.set_target_pose(T)

    # Gizmo → IK servo
    @ui.ik_target.on_update
    def _(event) -> None:
        if ui.is_manual_joint_mode():
            return
        T_target = ui.get_target_pose()
        q = solver.servo_step(solver.get_joint_state(), T_target)
        solver.set_joint_state(q)
        ui.sync_from_solver(solver)

    # ── Pose buttons ─────────────────────────────────────────────────

    def _go_to_joint_target(q_target: np.ndarray) -> None:
        """
        Move to a target joint configuration using joint-space interpolation.

        Joint values are interpolated directly from the current configuration
        to ``q_target``. The end-effector path is not explicitly constrained,
        so the Cartesian motion is generally curved.

        FK is used only to estimate a reasonable motion duration from the
        start and goal poses.

        Parameters
        ----------
        q_target : np.ndarray
            Target joint configuration in solver joint order.
        """
        def _run():
            with ui.busy():
                q_start = solver.get_joint_state()

                n_steps = compute_segment_steps_from_speed(
                    T_start= solver.fk(q_start),
                    T_goal=solver.fk(q_target),
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

    home_joint_btn = server.gui.add_button("Home (joint interp)")

    @home_joint_btn.on_click
    def _(event) -> None:
        _go_to_joint_target(Q_HOME)

    rest_joint_btn = server.gui.add_button("Rest (joint interp)")

    @rest_joint_btn.on_click
    def _(event) -> None:
        _go_to_joint_target(Q_REST)

    home_servo_btn = server.gui.add_button("Home (servo)")


    # ── Keep alive ────────────────────────────────────────────────────
    print("\n═══════════════════════════════════════════════")
    print("  robokin PyRoki + Viser — http://localhost:8080")
    print("  • Drag the gizmo to move the target")
    print("  • Toggle 'Manual joint control' for sliders")
    print("  • Ctrl+C to exit")
    print("═══════════════════════════════════════════════\n")

    server.sleep_forever()


if __name__ == "__main__":
    main()

