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

"""robokin.ui.viser_utils — pose ↔ Viser conversion helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to Viser ``(w, x, y, z)`` quaternion."""
    xyzw = Rotation.from_matrix(np.asarray(R, dtype=float)).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)


def wxyz_to_rotation_matrix(wxyz: np.ndarray) -> np.ndarray:
    """Convert a ``(w, x, y, z)`` quaternion to a 3×3 rotation matrix."""
    wxyz = np.asarray(wxyz, dtype=float)
    return Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()


def T_to_wxyz_xyz(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(wxyz, xyz)`` from a 4×4 homogeneous transform."""
    T = np.asarray(T, dtype=float)
    return rotation_matrix_to_wxyz(T[:3, :3]), np.array(T[:3, 3], dtype=float)


def wxyz_xyz_to_T(wxyz: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Build a 4×4 homogeneous transform from ``(wxyz, xyz)``."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = wxyz_to_rotation_matrix(wxyz)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def gizmo_to_T(gizmo) -> np.ndarray:
    """Read a Viser transform-controls handle into a 4×4 matrix."""
    return wxyz_xyz_to_T(np.array(gizmo.wxyz, dtype=float),
                         np.array(gizmo.position, dtype=float))


def T_to_gizmo(T: np.ndarray, gizmo) -> None:
    """Write a 4×4 matrix into a Viser transform-controls handle."""
    wxyz, xyz = T_to_wxyz_xyz(T)
    gizmo.wxyz = tuple(wxyz.tolist())
    gizmo.position = tuple(xyz.tolist())

