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

from roboplan.core import CartesianConfiguration, Scene
from roboplan.optimal_ik import (
    FrameTask,
    FrameTaskOptions,
    Oink,
    PositionLimit,
    VelocityLimit,
)

from .transformations import (
    compute_segment_steps_from_speed,
    ease_in_out_sine,
    interpolate_pose,
)

@dataclass(slots=True)
class OinkConfig:
    dt: float = 0.02
    position_cost: float = 1.0
    orientation_cost: float = 0.1
    task_gain: float = 1.0
    lm_damping: float = 0.01
    regularization: float = 1e-6
    use_velocity_limits: bool = True
    position_limit_gain: float = 1.0

    # Offline segment-generation defaults
    linear_speed_mps: float = 0.10
    angular_speed_radps: float = 1.0


class RoboPlanOinkKinematics:
    """Thin RoboPlan Scene + Oink wrapper for a robot arm."""

    def __init__(
        self,
        urdf_path: str | Path,
        srdf_path: str | Path,
        ee_frame: str,
        cfg: OinkConfig | None = None,
        *,
        group_name: str = "arm",
        base_frame: str = "base_link",
        scene_name: str = "oink_scene",
        yaml_config_path: str | Path | None = None,
        package_paths: list[str | Path] | None = None,
    ) -> None:
        # ── Setup ──
        self.urdf_path = Path(urdf_path)
        self.srdf_path = Path(srdf_path)
        self.yaml_config_path = yaml_config_path
        if isinstance(self.yaml_config_path, str):
            self.yaml_config_path = Path(self.yaml_config_path)

        self.ee_frame = ee_frame
        self.group_name = group_name
        self.base_frame = base_frame
        self.scene_name = scene_name
        self.cfg = cfg or OinkConfig()

        self.package_paths = [str(Path(p)) for p in (package_paths or [self.urdf_path.parent])]

        self.urdf_xml = self.urdf_path.read_text()
        srdf_xml = self.srdf_path.read_text()

        scene_args = {
            "urdf": self.urdf_xml,
            "srdf": srdf_xml,
            "package_paths": self.package_paths,
        }
        if self.yaml_config_path:
            scene_args["yaml_config_path"] = self.yaml_config_path
        self.scene = Scene(self.scene_name, **scene_args)

        self.group_info = self.scene.getJointGroupInfo(self.group_name)
        self.q_indices = np.array(self.group_info.q_indices, dtype=int)
        self.oink = Oink(self.scene, self.group_name)

        self.goal = None
        self.frame_task = None
        self._make_frame_task()

        self.constraints = [PositionLimit(self.oink, gain=self.cfg.position_limit_gain)]

        if self.cfg.use_velocity_limits:
            v_max = np.hstack(
                [
                    self.scene.getJointInfo(name).limits.max_velocity
                    for name in self.group_info.joint_names
                ]
            )
            self.constraints.append(VelocityLimit(self.oink, self.cfg.dt, v_max))

    def _make_frame_task(self) -> None:
        self.goal = CartesianConfiguration()
        self.goal.base_frame = self.base_frame
        self.goal.tip_frame = self.ee_frame

        task_options = FrameTaskOptions(
            position_cost=self.cfg.position_cost,
            orientation_cost=self.cfg.orientation_cost,
            task_gain=self.cfg.task_gain,
            lm_damping=self.cfg.lm_damping,
        )
        self.frame_task = FrameTask(self.oink, self.scene, self.goal, task_options)

    def _q_full_from_arm(self, q_arm: np.ndarray) -> np.ndarray:
        q_arm = np.asarray(q_arm, dtype=float)
        self._check_joint_vector(q_arm)

        q_full = self.scene.getCurrentJointPositions().copy()
        q_full[self.q_indices] = q_arm
        return q_full
    
    def set_joint_state(self, q_arm: np.ndarray) -> None:
        q_full = self._q_full_from_arm(q_arm)
        self.scene.setJointPositions(q_full)

    def get_joint_state(self) -> np.ndarray:
        q_full = self.scene.getCurrentJointPositions()
        return np.array(q_full[self.q_indices], dtype=float)

    def set_joint_state_deg(self, q_arm_deg: np.ndarray) -> None:
        self.set_joint_state(np.deg2rad(np.asarray(q_arm_deg, dtype=float)))

    def get_joint_state_deg(self) -> np.ndarray:
        return np.rad2deg(self.get_joint_state())
    
    def current_pose(self) -> np.ndarray:
        q_full = self.scene.getCurrentJointPositions()
        return np.array(self.scene.forwardKinematics(q_full, self.ee_frame), dtype=float)

    def fk(self, q_arm: np.ndarray) -> np.ndarray:
        q_arm = np.asarray(q_arm, dtype=float)
        self._check_joint_vector(q_arm)

        q_full = self.scene.getCurrentJointPositions().copy()
        q_full[self.q_indices] = q_arm
        return np.array(self.scene.forwardKinematics(q_full, self.ee_frame), dtype=float)

    def fk_deg(self, q_arm_deg: np.ndarray) -> np.ndarray:
        return self.fk(np.deg2rad(np.asarray(q_arm_deg, dtype=float)))
    
    def servo_step(self, q_measured: np.ndarray, T_ref: np.ndarray) -> np.ndarray:
        """
        One Oink differential IK step:
        - seed scene from measured/current joints
        - set frame target
        - solve one delta-q step
        - integrate and update scene
        - return next arm joint command
        """
        T_ref = self._check_transform(T_ref)

        q_current = self._q_full_from_arm(q_measured)
        self.scene.setJointPositions(q_current)

        self.frame_task.setTargetFrameTransform(T_ref)

        delta_q = np.zeros(self.nv, dtype=float)
        self.oink.solveIk(
            [self.frame_task],
            self.constraints,
            self.scene,
            delta_q,
            self.cfg.regularization,
        )

        q_next = self.scene.integrate(q_current, delta_q)
        self.scene.setJointPositions(q_next)
        
        # Keep FK current for next iteration / logging
        self.scene.forwardKinematics(q_next, self.ee_frame)

        return np.array(q_next[self.q_indices], dtype=float)

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
        Solve a fixed IK goal by repeatedly applying Oink differential IK.
        Returns the converged joint target in degrees.
        """
        T_goal = self._check_transform(T_goal)

        q_curr = np.asarray(q_seed, dtype=float).copy()
        self._check_joint_vector(q_curr)

        for _ in range(int(n_iters)):
            q_curr = self.servo_step(q_curr, T_goal)

        return q_curr
    
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
        Generate an offline segment by stepping Oink through interpolated
        Cartesian references.

        If n_steps is not provided, estimate it from Cartesian translation
        distance and cfg.linear_speed_mps.
        """
        T_goal = self._check_transform(T_goal)

        q_curr = np.asarray(q_start, dtype=float).copy()
        self._check_joint_vector(q_curr)
        T_start = self.fk(q_curr)

        if n_steps is None:
            n_steps = compute_segment_steps_from_speed(
                T_start=T_start,
                T_goal=T_goal,
                dt=self.cfg.dt,
                linear_speed_mps=self.cfg.linear_speed_mps,
                angular_speed_radps=self.cfg.angular_speed_radps,
                limit_peak_speed=True,
            )

        traj: list[np.ndarray] = []
        for k in range(1, n_steps + 1):
            alpha = ease_in_out_sine(k / n_steps)
            T_ref = interpolate_pose(T_start, T_goal, alpha)
            q_curr = self.servo_step(q_curr, T_ref)
            traj.append(q_curr.copy())

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
        if len(q) != len(self.q_indices):
            raise ValueError(f"Expected {len(self.q_indices)} joints, got {len(q)}")

    @staticmethod
    def _check_transform(T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"Expected transform with shape (4, 4), got {T.shape}")
        return T