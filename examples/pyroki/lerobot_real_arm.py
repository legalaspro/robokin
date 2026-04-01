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

"""SO-101 PyRoki + LeRobot — real-arm demo (single control loop).

All hardware I/O runs in one 50 Hz loop. Viser callbacks only update
desired state — no serial access from callbacks.

Modes:
- Manual joints: toggle "Manual override", move sliders
- Interactive IK: drag the EE gizmo
- Auto loop: click "Run loop" / "End loop"

Usage:
    python examples/pyroki/lerobot_real_arm.py
    python examples/pyroki/lerobot_real_arm.py --pid 15 0 5
    python examples/pyroki/lerobot_real_arm.py --port /dev/so101_follower
    python examples/pyroki/lerobot_real_arm.py --rerun
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import viser
import yourdfpy
from scipy.spatial.transform import Rotation

from robokin.pyroki import PyrokiKinematics, PyrokiConfig
from robokin.robot_model import load_robot_description
from robokin.transformations import (
    compute_segment_steps_from_speed,
    ease_in_out_sine,
    interpolate_pose,
)
from robokin.ui.viser_app import ViserRobotUI
from robokin.ui.rerun_logger import RerunRobotLogger

from helpers.so101_lerobot import (
    SO101LeRobotArm,
    SO101LeRobotConfig,
)


EE_LINK = "gripper_frame_link"
DT = 1.0 / 50.0
DWELL_TIME = 0.3


def make_pose(
    pos_mm: list[float] | np.ndarray,
    rotvec_rad: list[float] | np.ndarray,
) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec_rad, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(pos_mm, dtype=float) / 1000.0
    return T


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 PyRoki + LeRobot (control loop)")
    parser.add_argument("--pid", type=int, nargs=3, metavar=("P", "I", "D"), default=None)
    parser.add_argument("--port", type=str, default="/dev/so101_follower")
    parser.add_argument("--calibration-id", type=str, default="lerobot_follower_arm")
    parser.add_argument("--rerun", action="store_true", help="Enable Rerun logging")
    args = parser.parse_args()

    # ── Load robot model ──────────────────────────────────────────────
    model = load_robot_description("so_arm101_description")
    urdf_path = str(model.urdf_path)
    urdf = yourdfpy.URDF.load(urdf_path)

    # ── Solver ────────────────────────────────────────────────────────
    solver = PyrokiKinematics(
        urdf_path=urdf_path,
        ee_link_name=EE_LINK,
        cfg=PyrokiConfig(dt=DT),
    )

    # ── Hardware ──────────────────────────────────────────────────────
    arm = SO101LeRobotArm(
        SO101LeRobotConfig(port=args.port, calibration_id=args.calibration_id),
        solver_joint_names=solver.joint_names,
    )
    arm.connect()

    if args.pid is not None:
        p, i, d = args.pid
        print(f"Setting PID: P={p} I={i} D={d}")
        arm.set_pid_all(p, i, d)
    Q_REST = solver.make_configuration({
        "shoulder_pan": 0.0,
        "shoulder_lift": -np.pi / 2,
        "elbow_flex": np.pi / 2,
        "wrist_flex": np.deg2rad(42.97),
        "wrist_roll": 0.0,
    })

    # ── Warm up JAX ───────────────────────────────────────────────────
    print("Warming up JAX...")
    solver.warmup()
    print("JAX ready.")

    # ── Initial state ─────────────────────────────────────────────────
    q_init = arm.get_joint_state()
    solver.set_joint_state(q_init)
    T_init = solver.current_pose()

    # ── Preset poses ──────────────────────────────────────────────────
    T_home = solver.fk(solver.make_configuration({}))
    T_down = make_pose([136.4, 0.0, 62.0], [0.0, np.pi, 0.0])
    T_left = make_pose([136.4, 100.0, 62.0], [0.0, np.pi, 0.0])
    T_right = make_pose([136.4, -100.0, 62.0], [0.0, np.pi, 0.0])

    pose_loop_list: list[tuple[str, np.ndarray]] = [
        ("Home", T_home),
        ("Down", T_down),
        ("Left", T_left),
        ("Right", T_right),
    ]

    # ── Viser UI ──────────────────────────────────────────────────────
    server = viser.ViserServer()
    ui = ViserRobotUI(
        server=server,
        urdf=urdf,
        solver_joint_names=solver.joint_names,
        gripper_joint_name="gripper",
    )
    ui.build(
        initial_q=q_init,
        initial_T=T_init,
        enable_joint_sliders=True,
        enable_gripper=True,
        enable_gizmo=True,
    )

    loop_btn = server.gui.add_button("Run loop")
    rest_btn = server.gui.add_button("Rest")

    # ── Rerun logger ──────────────────────────────────────────────────
    rr_logger = None
    if args.rerun:
        rr_logger = RerunRobotLogger(
            urdf_path=urdf_path,
            joint_names=solver.joint_names,
            ee_frame=EE_LINK,
            app_id="so101_pyroki_real_arm",
        )
        rr_logger.init()

    # ══════════════════════════════════════════════════════════════════
    # Desired state — updated by callbacks, consumed by control loop
    # ══════════════════════════════════════════════════════════════════

    # Segment trajectory state
    segment_active = False
    segment_T_start: np.ndarray | None = None
    segment_T_goal: np.ndarray | None = None
    segment_t0: float = 0.0
    segment_duration: float = 0.0

    # Loop state
    loop_active = False
    loop_pose_idx = 0
    loop_dwelling = False
    loop_dwell_t0: float = 0.0

    # Pending gripper update
    gripper_dirty = False

    # ── Helpers to start/finish segments ──────────────────────────────

    def start_segment(T_goal: np.ndarray) -> None:
        nonlocal segment_active, segment_T_start, segment_T_goal, segment_t0, segment_duration
        q_meas = arm.get_joint_state()
        T_start = solver.fk(q_meas)
        n_steps = compute_segment_steps_from_speed(
            T_start=T_start,
            T_goal=T_goal,
            dt=DT,
            linear_speed_mps=solver.cfg.linear_speed_mps,
            angular_speed_radps=solver.cfg.angular_speed_radps,
        )
        segment_T_start = T_start
        segment_T_goal = T_goal
        segment_duration = n_steps * DT
        segment_t0 = time.perf_counter()
        segment_active = True
        ui.set_target_pose(T_goal)

    def finish_segment() -> None:
        nonlocal segment_active
        segment_active = False

    # ── Callbacks (only update desired state, no serial I/O) ─────────

    @ui.ik_target.on_update
    def _(event) -> None:
        pass  # gizmo target read directly in control loop

    if ui.gripper_slider is not None:
        @ui.gripper_slider.on_update
        def _(event) -> None:
            nonlocal gripper_dirty
            gripper_dirty = True

    @rest_btn.on_click
    def _(event) -> None:
        nonlocal loop_active
        loop_active = False
        loop_btn.label = "Run loop"
        T_rest = solver.fk(Q_REST)
        start_segment(T_rest)

    @loop_btn.on_click
    def _(event) -> None:
        nonlocal loop_active, loop_pose_idx, loop_dwelling
        if loop_active:
            # Stop
            loop_active = False
            loop_btn.label = "Run loop"
        else:
            # Start
            loop_active = True
            loop_pose_idx = 0
            loop_dwelling = False
            loop_btn.label = "End loop"
            name, T_goal = pose_loop_list[0]
            print(f"  loop -> {name}")
            start_segment(T_goal)

    # ══════════════════════════════════════════════════════════════════
    # Main control loop — single thread, fixed 50 Hz
    # ══════════════════════════════════════════════════════════════════

    print("\n═══════════════════════════════════════════════════════════════")
    print("  robokin PyRoki + LeRobot (control loop demo)")
    print("  Viser:  http://localhost:8080")
    print("═══════════════════════════════════════════════════════════════\n")

    try:
        while True:
            loop_t0 = time.perf_counter()

            # ── Read hardware ─────────────────────────────────────
            q_meas = arm.get_joint_state()

            # ── Gripper update ────────────────────────────────────
            if gripper_dirty and ui.gripper_slider is not None:
                arm.set_gripper_position(ui.gripper_slider.value)
                gripper_dirty = False

            # ── Compute command ───────────────────────────────────
            if segment_active:
                # Trajectory segment in progress
                elapsed = time.perf_counter() - segment_t0
                alpha = ease_in_out_sine(
                    1.0 if segment_duration <= 0 else elapsed / segment_duration
                )
                T_ref = interpolate_pose(segment_T_start, segment_T_goal, alpha)

                q_cmd = solver.solve_goal(q_meas, T_ref)
                solver.set_joint_state(q_cmd)
                arm.set_joint_state(q_cmd)
                ui.sync_from_solver(solver, move_gizmo=False)

                if elapsed >= segment_duration:
                    finish_segment()
                    ui.set_target_pose(segment_T_goal)

                    # Advance loop if active
                    if loop_active:
                        loop_dwelling = True
                        loop_dwell_t0 = time.perf_counter()

            elif loop_active and loop_dwelling:
                # Dwell between loop segments
                if time.perf_counter() - loop_dwell_t0 >= DWELL_TIME:
                    loop_dwelling = False
                    loop_pose_idx = (loop_pose_idx + 1) % len(pose_loop_list)
                    name, T_goal = pose_loop_list[loop_pose_idx]
                    print(f"  loop -> {name}")
                    start_segment(T_goal)

            elif ui.is_manual_joint_mode():
                # Manual slider mode
                q_cmd = ui.get_joint_values()
                solver.set_joint_state(q_cmd)
                T = solver.current_pose()
                ui.update_robot_from_joint_values(q_cmd)
                ui.update_ee_display(T)
                ui.set_target_pose(T)
                arm.set_joint_state(q_cmd)

            else:
                # Gizmo IK mode — servo toward gizmo target
                T_target = ui.get_target_pose()
                q_cmd = solver.servo_step(q_meas, T_target)
                solver.set_joint_state(q_cmd)
                arm.set_joint_state(q_cmd)
                ui.sync_from_solver(solver, move_gizmo=False)

            # ── Log to Rerun ──────────────────────────────────────
            if rr_logger is not None:
                rr_logger.log_state(solver.get_joint_state(), solver.current_pose())

            # ── Sleep to maintain frequency ───────────────────────
            sleep_left = DT - (time.perf_counter() - loop_t0)
            if sleep_left > 0:
                time.sleep(sleep_left)

    except KeyboardInterrupt:
        print("\nReturning to rest pose...")
        try:
            time.sleep(0.2)  # let serial port settle
            T_rest = solver.fk(Q_REST)
            q_meas = arm.get_joint_state()
            T_start = solver.fk(q_meas)
            n_steps = compute_segment_steps_from_speed(
                T_start=T_start, T_goal=T_rest, dt=DT,
                linear_speed_mps=solver.cfg.linear_speed_mps,
                angular_speed_radps=solver.cfg.angular_speed_radps,
            )
            duration = n_steps * DT
            t0 = time.perf_counter()
            while True:
                lt0 = time.perf_counter()
                elapsed = lt0 - t0
                alpha = ease_in_out_sine(1.0 if duration <= 0 else elapsed / duration)
                T_ref = interpolate_pose(T_start, T_rest, alpha)
                q_meas = arm.get_joint_state()
                q = solver.solve_goal(q_meas, T_ref)
                solver.set_joint_state(q)
                arm.set_joint_state(q)
                ui.sync_from_solver(solver, move_gizmo=False)
                if elapsed >= duration:
                    break
                sl = DT - (time.perf_counter() - lt0)
                if sl > 0:
                    time.sleep(sl)
        except ConnectionError as e:
            print(f"Recovery failed — serial error: {e}")
        except KeyboardInterrupt:
            print("Recovery interrupted — stopping immediately.")
    finally:
        try:
            arm.disconnect()
        except Exception as e:
            print(f"Warning: disconnect failed ({e})")


if __name__ == "__main__":
    main()
