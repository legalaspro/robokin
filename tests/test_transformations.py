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

"""Unit tests for robokin.transformations — pure math, no robot deps."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robokin.transformations import (
    ease_in_out_sine,
    interpolate_pose,
    pose_distance,
    compute_segment_steps_from_speed,
)


# ---------------------------------------------------------------------------
# ease_in_out_sine
# ---------------------------------------------------------------------------

def test_ease_boundary():
    assert ease_in_out_sine(0.0) == pytest.approx(0.0, abs=1e-12)
    assert ease_in_out_sine(1.0) == pytest.approx(1.0, abs=1e-12)


def test_ease_midpoint():
    assert ease_in_out_sine(0.5) == pytest.approx(0.5, abs=1e-12)


def test_ease_monotonic():
    vals = [ease_in_out_sine(t / 100.0) for t in range(101)]
    for a, b in zip(vals, vals[1:]):
        assert b >= a - 1e-15


# ---------------------------------------------------------------------------
# interpolate_pose
# ---------------------------------------------------------------------------

def _make_T(pos, rotvec=(0, 0, 0)):
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    return T


def test_interpolate_identity():
    T0 = _make_T([1, 0, 0])
    T1 = _make_T([3, 0, 0])
    T_mid = interpolate_pose(T0, T1, 0.5)
    np.testing.assert_allclose(T_mid[:3, 3], [2, 0, 0], atol=1e-10)


def test_interpolate_endpoints():
    T0 = _make_T([0, 0, 0], [0, 0, 0.5])
    T1 = _make_T([1, 2, 3], [0.3, 0.2, 0.1])
    np.testing.assert_allclose(interpolate_pose(T0, T1, 0.0), T0, atol=1e-10)
    np.testing.assert_allclose(interpolate_pose(T0, T1, 1.0), T1, atol=1e-10)


def test_interpolate_pose_rotation_is_orthonormal():
    T0 = _make_T([0, 0, 0], [0.0, 0.0, 0.0])
    T1 = _make_T([0, 0, 0], [0.3, -0.2, 0.4])
    T = interpolate_pose(T0, T1, 0.37)

    R = T[:3, :3]
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-8)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-8)


# ---------------------------------------------------------------------------
# pose_distance
# ---------------------------------------------------------------------------

def test_pose_distance_zero():
    T = _make_T([1, 2, 3], [0.1, 0.2, 0.3])
    dp, dw = pose_distance(T, T)
    assert dp == pytest.approx(0.0, abs=1e-10)
    assert dw == pytest.approx(0.0, abs=1e-10)


def test_pose_distance_translation():
    T0 = _make_T([0, 0, 0])
    T1 = _make_T([3, 4, 0])
    dp, dw = pose_distance(T0, T1)
    assert dp == pytest.approx(5.0, abs=1e-10)
    assert dw == pytest.approx(0.0, abs=1e-10)


def test_pose_distance_rotation():
    T0 = _make_T([0, 0, 0])
    T1 = _make_T([0, 0, 0], [0, 0, np.pi / 2])
    dp, dw = pose_distance(T0, T1)
    assert dp == pytest.approx(0.0, abs=1e-10)
    assert dw == pytest.approx(np.pi / 2, abs=1e-10)


# ---------------------------------------------------------------------------
# compute_segment_steps_from_speed
# ---------------------------------------------------------------------------

def test_segment_steps_basic():
    T0 = _make_T([0, 0, 0])
    T1 = _make_T([0.1, 0, 0])  # 100 mm
    dt = 0.02
    steps = compute_segment_steps_from_speed(
        T0, T1, dt,
        linear_speed_mps=0.1,
        angular_speed_radps=1.0,
        limit_peak_speed=False,
    )
    # distance=0.1m, speed=0.1m/s → duration=1s → steps=50
    assert steps == 50


def test_segment_steps_minimum():
    T = _make_T([0, 0, 0])
    steps = compute_segment_steps_from_speed(
        T, T, dt=0.02,
        linear_speed_mps=0.1,
        angular_speed_radps=1.0,
    )
    assert steps >= 1

