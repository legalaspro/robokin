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
Placo + Viser + Rerun — headless Cartesian servo preset loop.

Cycles through predefined poses using Cartesian servo (interpolating EE
pose and running servo_step at 50 Hz).  No buttons or interactive controls —
just launches and runs.

Viser:  3D robot view at http://localhost:8080
Rerun:  joint angles + EE position time-series

Usage:
    python examples/placo_viser_rerun_loop.py
"""

from __future__ import annotations

import time

import numpy as np
import viser
import yourdfpy
from scipy.spatial.transform import Rotation

from robokin.placo import PlacoKinematics, PlacoConfig
from robokin.robot_model import load_robot_description
from robokin.transformations import (
    compute_segment_steps_from_speed,
    ease_in_out_sine,
    interpolate_pose,
)
from robokin.ui.viser_app import ViserRobotUI
from robokin.ui.rerun_logger import RerunRobotLogger


# ── Robot config ──────────────────────────────────────────────────────

EE_FRAME = "gripper_frame_link"
DT = 1.0 / 50.0
DWELL_TIME = 0.3


# ── Pose helper ───────────────────────────────────────────────────────

def make_pose(
    pos_mm: list[float] | np.ndarray,
    rotvec_rad: list[float] | np.ndarray,
) -> np.ndarray:
    """Build a 4×4 pose from position in mm + rotation vector in rad."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec_rad, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(pos_mm, dtype=float) / 1000.0
    return T


def main() -> None:
    # ── Load robot ──
    model = load_robot_description("so_arm101_description")
    urdf_path = str(model.urdf_path)
    urdf = yourdfpy.URDF.load(urdf_path)

    # ── Create solver ──
    solver = PlacoKinematics(
        urdf_path=urdf_path,
        ee_frame=EE_FRAME,
        cfg=PlacoConfig(dt=DT),
    )

    # ── Preset poses ──
    T_home  = solver.fk(np.zeros(len(solver.joint_names), dtype=float))
    T_down  = make_pose([136.4,    0.0,  62.0], [0.0, 3.141, 0.0])
    T_left  = make_pose([136.4,  100.0,  62.0], [0.0, 3.141, 0.0])
    T_right = make_pose([136.4, -100.0,  62.0], [0.0, 3.141, 0.0])

    POSE_LOOP: list[tuple[str, np.ndarray]] = [
        ("Home",  T_home),
        ("Down",  T_down),
        ("Left",  T_left),
        ("Right", T_right),
    ]

    # ── Initial state ──
    q_init = solver.get_joint_state()
    T_init = solver.current_pose()

    # ── Viser UI (view-only: no sliders, no gizmo) ──
    server = viser.ViserServer()
    ui = ViserRobotUI(
        server=server, urdf=urdf,
        solver_joint_names=solver.joint_names,
        gripper_joint_name="gripper",
    )
    ui.build(
        initial_q=q_init, initial_T=T_init,
        enable_joint_sliders=False, enable_gripper=False, enable_gizmo=True,
    )

    # ── Rerun logger ──
    rr_logger = RerunRobotLogger(
        urdf_path=urdf_path,
        joint_names=solver.joint_names,
        ee_frame=EE_FRAME,
        app_id="placo_viser_rerun_loop",
    )
    rr_logger.init()

    # ── Cartesian servo segment runner ────────────────────────────────

    def run_segment(T_target: np.ndarray) -> None:
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
            q = solver.servo_step(solver.get_joint_state(), T_ref)
            solver.set_joint_state(q)
            ui.sync_from_solver(solver, move_gizmo=False)
            rr_logger.log_state(q, solver.current_pose())
            time.sleep(DT)

    # ── Main loop ─────────────────────────────────────────────────────

    print("\n═══════════════════════════════════════════════════════════════")
    print("  robokin Placo + Viser + Rerun (auto loop)")
    print("  Viser:  http://localhost:8080")
    print("  Rerun:  check terminal for viewer URL")
    print("  Ctrl+C to exit")
    print("═══════════════════════════════════════════════════════════════\n")

    time.sleep(1.0)  # let viewers connect

    try:
        while True:
            for name, T_target in POSE_LOOP:
                print(f"  → {name}")
                ui.set_target_pose(T_target)
                run_segment(T_target)
                time.sleep(DWELL_TIME)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

