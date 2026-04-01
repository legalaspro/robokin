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

import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
from pyroki.collision import HalfSpace, RobotCollision
import yourdfpy

from . import pyroki_snippets as pks

from .transformations import (
    compute_segment_steps_from_speed,
    ease_in_out_sine,
    interpolate_pose,
)

@dataclass(slots=True)
class PyrokiConfig:
    dt: float = 0.02
    mode: str = "vel_cost_collision"   # "basic_ik", "basic_ik_collision", "vel_cost_collision"
    vel_weight: float = 0.1
    linear_speed_mps: float = 0.10
    angular_speed_radps: float = 1.0

class PyrokiKinematics:
    """Thin PyRoki wrapper using full robot configuration vectors."""

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        ee_link_name: str = "",
        cfg: PyrokiConfig | None = None,
        *,
        urdf: yourdfpy.URDF | None = None,
    ) -> None:
        if urdf is not None:
            self.urdf = urdf
        elif urdf_path is not None:
            self.urdf = yourdfpy.URDF.load(str(urdf_path))
        else:
            raise ValueError("Provide either urdf_path or urdf")

        self.ee_link_name = ee_link_name
        self.cfg = cfg or PyrokiConfig()

        # ── PyRoki setup ──
        self.robot = pk.Robot.from_urdf(self.urdf)

        # End effector link index
        self.ee_link_index = self.robot.links.names.index(self.ee_link_name)

        # Build simple robot + ground collision world
        self.robot_coll = RobotCollision.from_urdf(self.urdf)
        self.world_coll = [
            HalfSpace.from_point_and_normal(
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.0], dtype=float),
            )
        ]

        # Canonical joint ordering (PyRoki topological sort — may differ
        # from the order joints appear in the URDF file).
        self.joint_names: list[str] = list(self.robot.joints.actuated_names)
        self._name_to_idx: dict[str, int] = {
            n: i for i, n in enumerate(self.joint_names)
        }

        # default full PyRoki joint configuration.
        self.current_cfg = self._zero_cfg()

    # ------------------------------------------------------------------
    # Joint helpers
    # ------------------------------------------------------------------

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    def _zero_cfg(self) -> np.ndarray:
        """Return the default full PyRoki joint configuration."""
        return np.array(self.robot.joint_var_cls.default_factory(), dtype=float)

    def make_configuration(self, joint_values: dict[str, float]) -> np.ndarray:
        """Build a full configuration vector from a ``{name: radians}`` dict.

        Joints not present in *joint_values* default to zero.
        """
        cfg = np.zeros(self.n_joints, dtype=float)
        for name, val in joint_values.items():
            if name not in self._name_to_idx:
                raise KeyError(
                    f"Unknown joint '{name}'. "
                    f"Known joints: {self.joint_names}"
                )
            cfg[self._name_to_idx[name]] = float(val)
        return cfg

    def warmup(self) -> None:
        """Run a dummy solve to trigger JAX JIT compilation."""
        T = self.current_pose()
        self.solve_goal(self.current_cfg.copy(), T)

    def set_joint_state(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float)
        self._check_cfg(q)
        self.current_cfg = q.copy()

    def get_joint_state(self) -> np.ndarray:
        return self.current_cfg.copy()

    def set_joint_state_deg(self, q_deg: np.ndarray) -> None:
        self.set_joint_state(np.deg2rad(np.asarray(q_deg, dtype=float)))

    def get_joint_state_deg(self) -> np.ndarray:
        return np.rad2deg(self.get_joint_state())
    
    def _ee_pose_parts(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return end-effector pose as (wxyz, xyz)."""
        q = np.asarray(q, dtype=float)
        self._check_cfg(q)

        Ts = self.robot.forward_kinematics(jnp.array(q))
        ee_se3 = jaxlie.SE3(Ts[self.ee_link_index])

        xyz = np.array(ee_se3.translation(), dtype=float)
        wxyz = np.array(ee_se3.rotation().wxyz, dtype=float)
        return wxyz, xyz
    
    def _pose_matrix_from_wxyz_xyz(self, wxyz: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        """Build a 4x4 pose matrix from PyRoki pose parts."""
        T = np.eye(4, dtype=float)
        T[:3, 3] = xyz

        # jaxlie gives wxyz; convert to rotation matrix through jaxlie itself
        R = np.array(jaxlie.SO3(wxyz=wxyz).as_matrix(), dtype=float)
        T[:3, :3] = R
        return T
    
    def _target_from_transform(self, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert a 4x4 transform to PyRoki target (position, wxyz)."""
        T = self._check_transform(T)

        xyz = np.array(T[:3, 3], dtype=float)
        wxyz = np.array(jaxlie.SO3.from_matrix(jnp.array(T[:3, :3])).wxyz, dtype=float)
        return xyz, wxyz

    def current_pose(self) -> np.ndarray:
        wxyz, xyz = self._ee_pose_parts(self.current_cfg)
        return self._pose_matrix_from_wxyz_xyz(wxyz, xyz)

    def fk(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        self._check_cfg(q)
        wxyz, xyz = self._ee_pose_parts(q)
        return self._pose_matrix_from_wxyz_xyz(wxyz, xyz)
    
    def fk_deg(self, q_deg: np.ndarray) -> np.ndarray:
        return self.fk(np.deg2rad(np.asarray(q_deg, dtype=float)))

    def _solve_once(
        self,
        q_seed: np.ndarray,
        T_goal: np.ndarray,
    ) -> np.ndarray:
        """Single IK solve. Returns the solved configuration."""
        q_seed = np.asarray(q_seed, dtype=float)
        self._check_cfg(q_seed)

        target_position, target_wxyz = self._target_from_transform(T_goal)

        if self.cfg.mode == "basic_ik":
            sol = pks.solve_ik(
                robot=self.robot,
                target_link_name=self.ee_link_name,
                target_position=target_position,
                target_wxyz=target_wxyz,
            )
        elif self.cfg.mode == "basic_ik_collision":
            sol = pks.solve_ik_with_collision(
                robot=self.robot,
                coll=self.robot_coll,
                world_coll_list=self.world_coll,
                target_link_name=self.ee_link_name,
                target_position=target_position,
                target_wxyz=target_wxyz,
            )
        elif self.cfg.mode == "vel_cost_collision":
            sol = pks.solve_ik_vel_cost_with_collision(
                robot=self.robot,
                coll=self.robot_coll,
                world_coll_list=self.world_coll,
                target_link_name=self.ee_link_name,
                target_position=target_position,
                target_wxyz=target_wxyz,
                prev_cfg=q_seed,
                dt=self.cfg.dt,
                vel_weight=self.cfg.vel_weight,
            )
        else:
            raise ValueError(f"Unsupported PyRoki mode: {self.cfg.mode}")

        self.current_cfg = np.asarray(sol, dtype=float)
        self._check_cfg(self.current_cfg)
        return self.current_cfg.copy()

    def servo_step(
        self,
        q_measured: np.ndarray,
        T_ref: np.ndarray,
    ) -> np.ndarray:
        """One IK solve step — use for real-time Cartesian servo."""
        return self._solve_once(q_measured, T_ref)

    def solve_goal(
        self,
        q_seed: np.ndarray,
        T_goal: np.ndarray,
    ) -> np.ndarray:
        """Solve IK for a goal pose, seeded from *q_seed*.

        Unlike ``servo_step`` there is no velocity cost, so the solver
        is free to jump to any reachable configuration.  Collision
        avoidance and joint limits are still enforced.
        """
        q_seed = np.asarray(q_seed, dtype=float)
        self._check_cfg(q_seed)
        target_position, target_wxyz = self._target_from_transform(T_goal)

        sol = pks.solve_ik(
                robot=self.robot,
                target_link_name=self.ee_link_name,
                target_position=target_position,
                target_wxyz=target_wxyz,
        )
        self.current_cfg = np.asarray(sol, dtype=float)
        self._check_cfg(self.current_cfg)
        return self.current_cfg.copy()

    def solve_goal_deg(
        self,
        q_seed_deg: np.ndarray,
        T_goal: np.ndarray,
    ) -> np.ndarray:
        q_sol = self.solve_goal(
            np.deg2rad(np.asarray(q_seed_deg, dtype=float)),
            T_goal,
        )
        return np.rad2deg(q_sol)
    
    def servo_step_deg(
        self,
        q_measured_deg: np.ndarray,
        T_ref: np.ndarray,
    ) -> np.ndarray:
        q_next = self.servo_step(
            np.deg2rad(np.asarray(q_measured_deg, dtype=float)),
            T_ref,
        )
        return np.rad2deg(q_next)

    def generate_segment(
        self,
        q_start: np.ndarray,
        T_goal: np.ndarray,
        n_steps: int | None = None,
    ) -> list[np.ndarray]:
        """
        Offline trajectory optimisation between current and goal EE poses.
        Returns a list of full configuration vectors.
        """
        q_curr = np.asarray(q_start, dtype=float).copy()
        self._check_cfg(q_curr)

        T_goal = self._check_transform(T_goal)
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

        start_position, start_wxyz = self._target_from_transform(T_start)
        end_position, end_wxyz = self._target_from_transform(T_goal)

        traj = pks.solve_trajopt(
            robot=self.robot,
            robot_coll=self.robot_coll,
            world_coll=self.world_coll,
            target_link_name=self.ee_link_name,
            start_position=start_position,
            start_wxyz=start_wxyz,
            end_position=end_position,
            end_wxyz=end_wxyz,
            timesteps=n_steps,
            dt=self.cfg.dt,
        )

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

    def _check_cfg(self, q: np.ndarray) -> None:
        if q.ndim != 1:
            raise ValueError(f"Expected 1D config vector, got shape {q.shape}")
        if q.shape != self.current_cfg.shape:
            raise ValueError(
                f"Expected config shape {self.current_cfg.shape}, got {q.shape}"
            )

    @staticmethod
    def _check_transform(T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"Expected transform with shape (4, 4), got {T.shape}")
        return T