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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import placo

from .transformations import (
    compute_segment_steps_from_speed,
    ease_in_out_sine,
    interpolate_pose,
)


@dataclass(slots=True)
class PlacoConfig:
    dt: float = 0.02
    pos_weight: float = 1.0
    rot_weight: float = 0.1
    internal_iters: int = 1
    regularization: float = 1e-4
    enable_velocity_limits: bool = True
    enable_self_collisions: bool = False
    self_collision_margin: float = 0.005
    self_collision_trigger: float = 0.01

    # Optional offline segment-generation defaults
    linear_speed_mps: float = 0.10
    angular_speed_radps: float = 1.0


class PlacoKinematics:
    """Thin Placo KinematicsSolver wrapper for a robot arm."""

    def __init__(
        self,
        urdf_path: str | Path,
        ee_frame: str,
        cfg: PlacoConfig | None = None,
    ) -> None:

        self.urdf_path = str(urdf_path)
        self.ee_frame = ee_frame
        self.cfg = cfg or PlacoConfig()

        # ── Placo setup ──
        self.robot = placo.RobotWrapper(self.urdf_path)
        self.joint_names = list(self.robot.joint_names())
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.solver.dt = self.cfg.dt
        
        # ── EE tracking task ──
        self.tip_task = self.solver.add_frame_task(self.ee_frame, np.eye(4))
        self.tip_task.configure(
            "ee",
            "soft",
            self.cfg.pos_weight,
            self.cfg.rot_weight,
        )
        
        # ── Regularization (penalises large joint velocities → smoother) ──
        self.reg_task = None
        if self.cfg.regularization > 0.0:
            self.reg_task = self.solver.add_regularization_task(self.cfg.regularization)

        # ── Velocity limits (caps joint speed per URDF limits) ──
        if self.cfg.enable_velocity_limits:
            self.solver.enable_velocity_limits(True)

        # ── Self-collision avoidance ──
        self.self_collision_constraint = None
        if self.cfg.enable_self_collisions:
            sc = self.solver.add_avoid_self_collisions_constraint()
            sc.self_collisions_margin = self.cfg.self_collision_margin
            sc.self_collisions_trigger = self.cfg.self_collision_trigger
            self.self_collision_constraint = sc

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    def make_configuration(self, joint_values: dict[str, float]) -> np.ndarray:
        """Build a configuration vector from a ``{name: radians}`` dict.

        Joints not present in *joint_values* default to zero.
        """
        name_to_idx = {n: i for i, n in enumerate(self.joint_names)}
        cfg = np.zeros(self.n_joints, dtype=float)
        for name, val in joint_values.items():
            if name not in name_to_idx:
                raise KeyError(
                    f"Unknown joint '{name}'. "
                    f"Known joints: {self.joint_names}"
                )
            cfg[name_to_idx[name]] = float(val)
        return cfg

    def set_joint_state(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float)
        self._check_joint_vector(q)

        for i, name in enumerate(self.joint_names):
            self.robot.set_joint(name, float(q[i]))
        self.robot.update_kinematics()

    def get_joint_state(self) -> np.ndarray:
        return np.array(
            [self.robot.get_joint(name) for name in self.joint_names],
            dtype=float,
        )

    def set_joint_state_deg(self, q_deg: np.ndarray) -> None:
        self.set_joint_state(np.deg2rad(np.asarray(q_deg, dtype=float)))

    def get_joint_state_deg(self) -> np.ndarray:
        return np.rad2deg(self.get_joint_state())

    def current_pose(self) -> np.ndarray:
        self.robot.update_kinematics()
        return np.array(self.robot.get_T_world_frame(self.ee_frame), dtype=float)

    def fk(self, q: np.ndarray) -> np.ndarray:
        self.set_joint_state(q)
        return self.current_pose().copy()

    def fk_deg(self, q_deg: np.ndarray) -> np.ndarray:
        return self.fk(np.deg2rad(np.asarray(q_deg, dtype=float)))

    def servo_step(self, q_measured: np.ndarray, T_ref: np.ndarray) -> np.ndarray:
        """
        One differential IK step:
        - seed solver from measured/current joints
        - update frame target
        - solve a small fixed number of internal iterations
        - return next joint command in degrees
        """
        T_ref = self._check_transform(T_ref)
        self.set_joint_state(q_measured)
        self.tip_task.T_world_frame = T_ref

        for _ in range(self.cfg.internal_iters):
            self.solver.solve(True)
            self.robot.update_kinematics()

        return self.get_joint_state()

    def servo_step_deg(self, q_measured_deg: np.ndarray, T_ref: np.ndarray) -> np.ndarray:
        q_next = self.servo_step(np.deg2rad(np.asarray(q_measured_deg, dtype=float)), T_ref)
        return np.rad2deg(q_next)

    def solve_goal(
        self,
        q_seed: np.ndarray,
        T_goal: np.ndarray,
        n_iters: int = 100,
    ) -> np.ndarray:
        """
        Solve a fixed IK goal by iterating from a seed.
        Returns the converged joint target in degrees.
        """
        T_goal = self._check_transform(T_goal)
        self.set_joint_state(q_seed)
        self.tip_task.T_world_frame = T_goal

        for _ in range(int(n_iters)):
            self.solver.solve(True)
            self.robot.update_kinematics()

        return self.get_joint_state()
    
    def solve_goal_deg(
        self,
        q_seed_deg: np.ndarray,
        T_goal: np.ndarray,
        n_iters: int = 100,
    ) -> np.ndarray:
        q_sol = self.solve_goal(
            np.deg2rad(np.asarray(q_seed_deg, dtype=float)),
            T_goal,
            n_iters=n_iters,
        )
        return np.rad2deg(q_sol)

    def generate_segment(
        self,
        q_start: np.ndarray,
        T_goal: np.ndarray,
        n_steps: int | None = None,
    ) -> list[np.ndarray]:
        """
        Offline waypoint segment generation.

        Uses previous command as the next measured state.
        For real hardware closed-loop execution, replace this with fresh joint
        measurements each control tick.
        """
        T_goal = self._check_transform(T_goal)
        q_meas = np.asarray(q_start, dtype=float).copy()
        self._check_joint_vector(q_meas)

        T_start = self.fk(q_meas)

        if n_steps is None:
            n_steps = compute_segment_steps_from_speed(
                T_start=T_start,
                T_goal=T_goal,
                dt=self.cfg.dt,
                linear_speed_mps=self.cfg.linear_speed_mps,
                angular_speed_radps=self.cfg.angular_speed_radps,
                limit_peak_speed=True,
            )
        else:
            n_steps = int(n_steps)

        traj: list[np.ndarray] = []
        for k in range(1, n_steps + 1):
            alpha = ease_in_out_sine(k / n_steps)
            T_ref = interpolate_pose(T_start, T_goal, alpha)
            q_cmd = self.servo_step(q_meas, T_ref)
            traj.append(q_cmd.copy())
            q_meas = q_cmd

        return traj
    
    def generate_segment_deg(
        self,
        q_start_deg: np.ndarray,
        T_goal: np.ndarray,
        n_steps: int | None = None,
    ) -> list[np.ndarray]:
        traj = self.generate_segment(
            np.deg2rad(np.asarray(q_start_deg, dtype=float)),
            T_goal,
            n_steps=n_steps,
        )
        return [np.rad2deg(q) for q in traj]

    def _check_joint_vector(self, q: np.ndarray) -> None:
        if q.ndim != 1:
            raise ValueError(f"Expected 1D joint vector, got shape {q.shape}")
        if len(q) != len(self.joint_names):
            raise ValueError(f"Expected {len(self.joint_names)} joints, got {len(q)}")

    @staticmethod
    def _check_transform(T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"Expected transform with shape (4, 4), got {T.shape}")
        return T


