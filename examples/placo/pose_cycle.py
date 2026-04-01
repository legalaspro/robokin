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
Placo + Viser preset loop example using robokin.

Cycles through a list of predefined poses using two motion strategies:

  • Joint interp:    solve IK once, then interpolate joints directly
  • Cartesian servo: interpolate EE pose and run servo_step at 50 Hz

Each loop button toggles start/stop for its mode.

Usage:
    python examples/placo_viser_preset_loop.py
"""

from __future__ import annotations

import threading
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


# ── Robot config ──────────────────────────────────────────────────────

EE_FRAME = "gripper_frame_link"
DT = 1.0 / 50.0
DWELL_TIME = 0.25


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

    # ── Viser UI (no sliders, no gizmo) ──
    server = viser.ViserServer()
    ui = ViserRobotUI(server=server, urdf=urdf,
                      solver_joint_names=solver.joint_names,
                      gripper_joint_name="gripper")
    ui.build(initial_q=q_init, initial_T=T_init,
             enable_joint_sliders=False, enable_gripper=False,
             enable_gizmo=True)


    # ── Optional status ───────────────────────────────────────────────
    status_handle = server.gui.add_text("Motion status", initial_value="idle")

    # ── Runner state ──
    loop_lock = threading.Lock()
    loop_stop_event: threading.Event | None = None
    loop_thread: threading.Thread | None = None
    active_loop_mode: str | None = None  # "joint", "servo", or None
    
    
     # ── Motion primitives ─────────────────────────────────────────────

    def _run_segment_joint_interp(
        T_target: np.ndarray,
        stop_event: threading.Event | None = None,
    ) -> bool:
        """
        Move to T_target by solving IK once, then interpolating in joint space.

        Returns False if interrupted by stop_event.
        """
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
            if stop_event is not None and stop_event.is_set():
                return False

            alpha = ease_in_out_sine(k / n_steps)
            q = q_start + alpha * (q_target - q_start)
            solver.set_joint_state(q)
            ui.sync_from_solver(solver, move_gizmo=False)
            time.sleep(DT)

        return True

    def _run_segment_cartesian_servo(
        T_target: np.ndarray,
        stop_event: threading.Event | None = None,
    ) -> bool:
        """
        Move to T_target by interpolating end-effector poses and running
        servo_step at each control tick.

        Returns False if interrupted by stop_event.
        """
        T_start = solver.current_pose()
        n_steps = compute_segment_steps_from_speed(
            T_start=T_start,
            T_goal=T_target,
            dt=DT,
            linear_speed_mps=solver.cfg.linear_speed_mps,
            angular_speed_radps=solver.cfg.angular_speed_radps,
        )

        for k in range(1, n_steps + 1):
            if stop_event is not None and stop_event.is_set():
                return False

            alpha = ease_in_out_sine(k / n_steps)
            T_ref = interpolate_pose(T_start, T_target, alpha)
            q = solver.servo_step(solver.get_joint_state(), T_ref)
            solver.set_joint_state(q)
            ui.sync_from_solver(solver, move_gizmo=False)
            time.sleep(DT)
        return True

    # ── Loop control helpers ──────────────────────────────────────────

    def _run_pose_loop(mode: str, stop_event: threading.Event) -> None:
        nonlocal loop_thread, loop_stop_event, active_loop_mode

        runner = (
            _run_segment_joint_interp
            if mode == "joint"
            else _run_segment_cartesian_servo
        )

        status_handle.value = f"loop running: {mode}"

        try:
            with ui.busy():
                while not stop_event.is_set():
                    for name, T_target in POSE_LOOP:
                        if stop_event.is_set():
                            return

                        status_handle.value = f"loop running: {mode} → {name}"
                        ui.set_target_pose(T_target)

                        ok = runner(T_target, stop_event)
                        if not ok:
                            return

                        # small dwell at each target
                        time.sleep(DWELL_TIME)
        finally:
            with loop_lock:
                # Clear only if this worker still owns the active stop_event.
                if loop_stop_event is stop_event:
                    loop_thread = None
                    loop_stop_event = None
                    active_loop_mode = None
            status_handle.value = "idle"

    def _toggle_loop(mode: str) -> None:
        """
        Toggle a preset loop.
        - click current mode again -> stop
        - click other mode -> stop old loop, start new one
        """
        nonlocal loop_thread, loop_stop_event, active_loop_mode

        with loop_lock:
            current_mode = active_loop_mode
            current_thread = loop_thread
            current_stop_event = loop_stop_event

        # Same button clicked again -> stop current loop.
        if current_mode == mode and current_stop_event is not None:
            current_stop_event.set()
            return

        # Different loop active -> stop it first.
        if current_stop_event is not None:
            current_stop_event.set()
        if current_thread is not None and current_thread.is_alive():
            current_thread.join(timeout=2.0)

        new_stop_event = threading.Event()
        new_thread = threading.Thread(
            target=_run_pose_loop,
            args=(mode, new_stop_event),
            daemon=True,
        )

        with loop_lock:
            loop_thread = new_thread
            loop_stop_event = new_stop_event
            active_loop_mode = mode

        new_thread.start()


    with server.gui.add_folder("Preset loop"):
        run_loop_joint_btn = server.gui.add_button("Run preset loop (joint interp)")
        run_loop_servo_btn = server.gui.add_button("Run preset loop (pose servo)")

    @run_loop_joint_btn.on_click
    def _(event) -> None:
        _toggle_loop("joint")

    @run_loop_servo_btn.on_click
    def _(event) -> None:
        _toggle_loop("servo")


    # ── Keep alive ──
    print("\n═══════════════════════════════════════════════════════════════")
    print("  robokin Placo + Viser (preset loop) — http://localhost:8080")
    print("  • Press button to start/stop cycling through poses")
    print("  • Ctrl+C to exit")
    print("═══════════════════════════════════════════════════════════════\n")

    server.sleep_forever()


if __name__ == "__main__":
    main()

