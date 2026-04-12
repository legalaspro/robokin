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
    """Sinusoidal ease-in / ease-out in [0, 1].  C1 continuous."""
    return 0.5 * (1.0 - math.cos(math.pi * float(t)))


def ease_quintic(t: float) -> float:
    """Quintic ease in [0, 1].  C2 continuous (zero vel + accel at boundaries)."""
    s = float(t)
    return 6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3


# Peak-velocity / average-velocity ratio for each easing, used by
# compute_segment_steps_from_speed to convert desired peak speed → duration.
EASING_PEAK_FACTOR: dict[str, float] = {
    "sine": math.pi / 2.0,      # ≈ 1.571
    "quintic": 15.0 / 8.0,      # = 1.875
}


def get_easing_fn(name: str = "quintic"):
    """Return an easing function by name."""
    if name == "sine":
        return ease_in_out_sine
    if name == "quintic":
        return ease_quintic
    raise ValueError(f"Unknown easing: {name!r}. Choose 'sine' or 'quintic'.")


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
# Pose construction
# ---------------------------------------------------------------------------

def make_pose(xyz, rotvec_rad) -> np.ndarray:
    """Build a 4x4 homogeneous transform from position (meters) and rotation vector."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec_rad, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(xyz, dtype=float)
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
    easing: str = "sine",
) -> int:
    """
    Estimate number of interpolation steps from Cartesian distance and rotation.

    If *limit_peak_speed* is True the provided speeds are treated as peak
    speeds for the eased trajectory.  The *easing* name selects the
    corresponding peak-factor (``"sine"`` or ``"quintic"``).
    """
    dpos, dang = pose_distance(T_start, T_goal)

    easing_factor = EASING_PEAK_FACTOR.get(easing, math.pi / 2.0) if limit_peak_speed else 1.0

    t_pos = 0.0 if linear_speed_mps <= 0 else easing_factor * dpos / linear_speed_mps
    t_rot = 0.0 if angular_speed_radps <= 0 else easing_factor * dang / angular_speed_radps

    duration = max(t_pos, t_rot, dt)
    return max(1, math.ceil(duration / dt))
