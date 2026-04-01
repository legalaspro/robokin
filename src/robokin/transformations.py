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

"""Transformation / interpolation utilities used by all kinematics backends."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


# ---------------------------------------------------------------------------
# Easing
# ---------------------------------------------------------------------------

def ease_in_out_sine(t: float) -> float:
    """Sinusoidal ease-in / ease-out in [0, 1]."""
    return 0.5 * (1.0 - math.cos(math.pi * float(t)))


# ---------------------------------------------------------------------------
# SE(3) interpolation  (scipy only — no pinocchio dependency)
# ---------------------------------------------------------------------------

def interpolate_pose(
    T_start: np.ndarray,
    T_goal: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """SE(3) interpolation: linear translation + SLERP rotation."""
    alpha = float(np.clip(alpha, 0.0, 1.0))

    T = np.eye(4, dtype=float)
    T[:3, 3] = (1.0 - alpha) * T_start[:3, 3] + alpha * T_goal[:3, 3]

    R_start = Rotation.from_matrix(T_start[:3, :3])
    R_goal = Rotation.from_matrix(T_goal[:3, :3])
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([R_start, R_goal]))
    T[:3, :3] = slerp([alpha])[0].as_matrix()
    return T


# ---------------------------------------------------------------------------
# Pose distance
# ---------------------------------------------------------------------------

def pose_distance(
    T_a: np.ndarray,
    T_b: np.ndarray,
) -> tuple[float, float]:
    """Return (translation distance [m], rotation distance [rad])."""
    dp = float(np.linalg.norm(T_b[:3, 3] - T_a[:3, 3]))

    R_rel = Rotation.from_matrix(T_a[:3, :3].T @ T_b[:3, :3])
    dw = float(np.linalg.norm(R_rel.as_rotvec()))
    return dp, dw


# ---------------------------------------------------------------------------
# Segment step estimation
# ---------------------------------------------------------------------------

def compute_segment_steps_from_speed(
    T_start: np.ndarray,
    T_goal: np.ndarray,
    dt: float,
    linear_speed_mps: float,
    angular_speed_radps: float,
    limit_peak_speed: bool = True,
) -> int:
    """
    Estimate number of interpolation steps from Cartesian distance and rotation.

    If *limit_peak_speed* is True the provided speeds are treated as peak
    speeds for the sinusoidal-eased trajectory.
    """
    dpos, dang = pose_distance(T_start, T_goal)

    easing_factor = (math.pi / 2.0) if limit_peak_speed else 1.0

    t_pos = 0.0 if linear_speed_mps <= 0 else easing_factor * dpos / linear_speed_mps
    t_rot = 0.0 if angular_speed_radps <= 0 else easing_factor * dang / angular_speed_radps

    duration = max(t_pos, t_rot, dt)
    return max(1, math.ceil(duration / dt))
