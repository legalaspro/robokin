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
Placo + Viser interactive IK example using robokin.

Demonstrates ViserRobotUI + PlacoKinematics with callback-driven events.
The gizmo is tracked with differential IK, and named-pose buttons move
the robot by generating a Cartesian end-effector segment to the target pose.

No hardware required.

Usage:
    python examples/placo_viser.py
"""

from __future__ import annotations

import threading
import time

import numpy as np
import viser
import yourdfpy

from robokin.placo import PlacoKinematics, PlacoConfig
from robokin.robot_model import load_robot_description
from robokin.ui.viser_app import ViserRobotUI


# ── Robot config ──────────────────────────────────────────────────────

EE_FRAME = "gripper_frame_link"
DT = 1.0 / 50.0


def main() -> None:
    # ── Load robot model ──
    model = load_robot_description("so_arm101_description")
    urdf_path = str(model.urdf_path)
    urdf = yourdfpy.URDF.load(urdf_path)

    # ── Create solver ──
    solver = PlacoKinematics(
        urdf_path=urdf_path,
        ee_frame=EE_FRAME,
        cfg=PlacoConfig(dt=DT),
    )

    # ── Named poses (built from joint names → order-independent) ──
    Q_HOME = solver.make_configuration({})
    Q_REST = solver.make_configuration({
        "shoulder_pan":  0.0,
        "shoulder_lift": -np.pi / 2,
        "elbow_flex":    np.pi / 2,
        "wrist_flex":    np.deg2rad(42.97),
        "wrist_roll":    0.0,
    })

    # ── Initial state ──
    solver.set_joint_state(Q_REST)
    T_init = solver.current_pose()
    q_init = solver.get_joint_state()

    # ── Viser UI ──
    server = viser.ViserServer()
    ui = ViserRobotUI(server=server, urdf=urdf,
                      solver_joint_names=solver.joint_names,
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
    if ui.gripper_slider is not None:
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

    def _go_to_pose(q_target: np.ndarray) -> None:
        """
        Move to the pose of ``q_target`` using an offline Cartesian servo segment.

        ``q_target`` is converted to a goal end-effector pose with FK, then
        Placo generates a pose-interpolated IK trajectory from the current
        state to that pose.
        """
        def _run():
            with ui.busy():
                q_start = solver.get_joint_state()   # capture before fk() clobbers state
                T_goal = solver.fk(q_target)
                traj = solver.generate_segment(q_start, T_goal)
                for q in traj:
                    solver.set_joint_state(q)
                    ui.sync_from_solver(solver)
                    time.sleep(DT)
        threading.Thread(target=_run, daemon=True).start()

    home_btn = server.gui.add_button("Home")

    @home_btn.on_click
    def _(event) -> None:
        _go_to_pose(Q_HOME)

    rest_btn = server.gui.add_button("Rest")

    @rest_btn.on_click
    def _(event) -> None:
        _go_to_pose(Q_REST)

    # ── Keep alive ────────────────────────────────────────────────────
    print("\n═══════════════════════════════════════════════")
    print("  robokin Placo + Viser — http://localhost:8080")
    print("  • Drag the gizmo to move the target")
    print("  • Toggle 'Manual joint control' for sliders")
    print("  • Ctrl+C to exit")
    print("═══════════════════════════════════════════════\n")

    server.sleep_forever()


if __name__ == "__main__":
    main()