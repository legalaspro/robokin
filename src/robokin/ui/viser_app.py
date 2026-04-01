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

from contextlib import contextmanager

import numpy as np
import viser
import yourdfpy
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

from robokin.ui.viser_utils import T_to_wxyz_xyz, gizmo_to_T, T_to_gizmo


class ViserRobotUI:
    """Minimal Viser helper — builds widgets, exposes handles, no behavior.

    The caller attaches callbacks (``on_click``, ``on_update``) to the
    public handles and wires them to their own solver / hardware logic.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        urdf: yourdfpy.URDF,
        *,
        solver_joint_names: list[str] | None = None,
        gripper_joint_name: str | None = None,
        root_node_name: str = "/robot",
        target_scale: float = 0.15,
        show_grid: bool = True,
    ) -> None:
        self.server = server
        self.urdf = urdf
        # Viser / yourdfpy canonical order — used for update_cfg()
        self._viser_joint_names = list(self.urdf.actuated_joint_names)
        # Solver order — the order q vectors arrive in from the solver.
        # If not provided, assume it matches Viser order.
        self._solver_joint_names = solver_joint_names or list(self._viser_joint_names)
        # Build reorder maps: solver index → viser index and vice versa
        self._solver_to_viser = [
            self._viser_joint_names.index(n) for n in self._solver_joint_names
        ]
        self._viser_to_solver = [
            self._solver_joint_names.index(n) for n in self._viser_joint_names
        ]
        self.gripper_joint_name = gripper_joint_name
        self.root_node_name = root_node_name
        self.target_scale = float(target_scale)

        if show_grid:
            self.server.scene.add_grid("/ground", width=2.0, height=2.0,
                                       cell_size=0.1)

        self.urdf_vis = ViserUrdf(self.server, self.urdf,
                                  root_node_name=self.root_node_name)

        # Public handles — populated by build()
        self.ik_target = None
        self.mode_toggle = None
        self.ee_pos_handle = None
        self.ee_rpy_handle = None
        self.joint_sliders: dict[str, object] = {}
        self.gripper_slider = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        initial_q: np.ndarray | None = None,
        initial_T: np.ndarray | None = None,
        enable_gizmo: bool = True,
        enable_joint_sliders: bool = False,
        enable_gripper: bool = False,
    ) -> None:
        if initial_q is None:
            initial_q = np.zeros(len(self._solver_joint_names), dtype=float)
        else:
            initial_q = np.asarray(initial_q, dtype=float)

        if initial_T is None:
            initial_T = np.eye(4, dtype=float)
        else:
            initial_T = np.asarray(initial_T, dtype=float)

        self.update_robot_from_joint_values(initial_q)

        # Target gizmo
        if enable_gizmo:
            wxyz, xyz = T_to_wxyz_xyz(initial_T)
            self.ik_target = self.server.scene.add_transform_controls(
                "/ik_target",
                scale=self.target_scale,
                position=tuple(xyz.tolist()),
                wxyz=tuple(wxyz.tolist()),
            )

        # EE display
        with self.server.gui.add_folder("EE Pose"):
            self.ee_pos_handle = self.server.gui.add_vector3(
                "Position (m)",
                initial_value=tuple(initial_T[:3, 3].tolist()),
                step=0.001, disabled=True,
            )
            rpy = Rotation.from_matrix(initial_T[:3, :3]).as_euler("xyz")
            self.ee_rpy_handle = self.server.gui.add_vector3(
                "RPY (rad)",
                initial_value=tuple(np.round(rpy, 4).tolist()),
                step=0.01, disabled=True,
            )

        # Joint sliders with manual override toggle
        arm_sliders: dict[str, object] = {}
        with self.server.gui.add_folder("Joints"):
            self.mode_toggle = self.server.gui.add_checkbox(
                "Manual override", initial_value=False, disabled=not enable_joint_sliders
            )
            for idx, jname in enumerate(self._solver_joint_names):
                if jname == self.gripper_joint_name:
                    continue  # gripper gets its own section
                joint = self.urdf.joint_map[jname]
                if joint.type not in ("revolute", "continuous", "prismatic"):
                    continue
                lo = float(joint.limit.lower) if joint.limit else -3.14
                hi = float(joint.limit.upper) if joint.limit else 3.14
                slider = self.server.gui.add_slider(
                    jname, min=lo, max=hi, step=0.01,
                    initial_value=float(initial_q[idx]),
                    disabled=True,
                )
                self.joint_sliders[jname] = slider
                arm_sliders[jname] = slider

            # Wire mode toggle → arm slider disabled state
            @self.mode_toggle.on_update
            def _toggle_sliders(event) -> None:
                manual = bool(self.mode_toggle.value)
                for slider in arm_sliders.values():
                    slider.disabled = not manual

        # Gripper — always editable, separate section
        if enable_gripper and self.gripper_joint_name is not None:
            grip_idx = self._solver_joint_names.index(self.gripper_joint_name)
            joint = self.urdf.joint_map[self.gripper_joint_name]
            lo = float(joint.limit.lower) if joint.limit else -3.14
            hi = float(joint.limit.upper) if joint.limit else 3.14
            self.gripper_slider = self.server.gui.add_slider(
                "Gripper", min=lo, max=hi, step=0.01,
                initial_value=float(initial_q[grip_idx]),
            )

    # ------------------------------------------------------------------
    # Reorder helpers  (solver order ↔ viser order)
    # ------------------------------------------------------------------

    def _solver_to_viser_q(self, q_solver: np.ndarray) -> np.ndarray:
        """Reorder a q vector from solver joint order to Viser joint order."""
        q_solver = np.asarray(q_solver, dtype=float)
        # _viser_to_solver[viser_idx] = solver_idx → gather from solver
        return q_solver[self._viser_to_solver]

    def _viser_to_solver_q(self, q_viser: np.ndarray) -> np.ndarray:
        """Reorder a q vector from Viser joint order to solver joint order."""
        q_viser = np.asarray(q_viser, dtype=float)
        # _solver_to_viser[solver_idx] = viser_idx → gather from viser
        return q_viser[self._solver_to_viser]

    # ------------------------------------------------------------------
    # Mode query
    # ------------------------------------------------------------------

    def is_manual_joint_mode(self) -> bool:
        return bool(self.mode_toggle.value) if self.mode_toggle is not None else False

    # ------------------------------------------------------------------
    # Busy guard — disable controls during trajectory execution
    # ------------------------------------------------------------------

    def set_controls_enabled(self, enabled: bool) -> None:
        """Enable or disable all interactive controls (gizmo, sliders, buttons).
        Gripper slider is also disabled while busy."""
        manual = self.is_manual_joint_mode()
        for slider in self.joint_sliders.values():
            slider.disabled = not enabled or not manual
        if self.gripper_slider is not None:
            self.gripper_slider.disabled = not enabled 
        if self.ik_target is not None:
            self.ik_target.disabled = not enabled or manual

    @contextmanager
    def busy(self):
        """Context manager that disables controls for the duration."""
        self.set_controls_enabled(False)
        try:
            yield
        finally:
            self.set_controls_enabled(True)

    # ------------------------------------------------------------------
    # Target gizmo helpers
    # ------------------------------------------------------------------

    def get_target_pose(self) -> np.ndarray:
        if self.ik_target is None:
            raise RuntimeError("Target gizmo not created — call build() first.")
        return gizmo_to_T(self.ik_target)

    def set_target_pose(self, T: np.ndarray) -> None:
        if self.ik_target is None:
            raise RuntimeError("Target gizmo not created — call build() first.")
        T_to_gizmo(np.asarray(T, dtype=float), self.ik_target)

    # ------------------------------------------------------------------
    # Joint slider helpers
    # ------------------------------------------------------------------

    def get_joint_values(self) -> np.ndarray:
        vals = []
        for name in self._solver_joint_names:
            if name == self.gripper_joint_name and self.gripper_slider is not None:
                vals.append(float(self.gripper_slider.value))
            elif name in self.joint_sliders:
                vals.append(float(self.joint_sliders[name].value))
        return np.array(vals, dtype=float)

    def set_joint_values(self, q_rad: np.ndarray) -> None:
        """Set arm joint sliders. Gripper is never overwritten — it is
        independently controlled via ``gripper_slider``."""
        q_rad = np.asarray(q_rad, dtype=float)
        for i, name in enumerate(self._solver_joint_names):
            if name == self.gripper_joint_name:
                continue  # gripper is user-controlled, skip
            if name in self.joint_sliders:
                self.joint_sliders[name].value = float(q_rad[i])

    # ------------------------------------------------------------------
    # Robot visualization
    # ------------------------------------------------------------------

    def update_robot_from_joint_values(self, q_rad: np.ndarray) -> None:
        """Update the 3D visualization.

        *q_rad* is in **solver** joint order.  It is reordered internally
        to match the Viser / yourdfpy order before calling ``update_cfg``.
        The gripper value is always read from its slider so that IK
        updates never override the user's setting.
        """
        q_rad = np.array(q_rad, dtype=float)
        if self.gripper_joint_name is not None and self.gripper_slider is not None:
            idx = self._solver_joint_names.index(self.gripper_joint_name)
            q_rad[idx] = float(self.gripper_slider.value)
        # update_cfg expects viser (yourdfpy) joint order
        q_viser = self._solver_to_viser_q(q_rad)
        self.urdf_vis.update_cfg(q_viser)

    # ------------------------------------------------------------------
    # EE display
    # ------------------------------------------------------------------

    def update_ee_display(self, T: np.ndarray) -> None:
        T = np.asarray(T, dtype=float)
        if self.ee_pos_handle is not None:
            self.ee_pos_handle.value = tuple(T[:3, 3].tolist())
        if self.ee_rpy_handle is not None:
            rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
            self.ee_rpy_handle.value = tuple(np.round(rpy, 4).tolist())

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------

    def sync_from_solver(self, solver, move_gizmo: bool = True) -> None:
        q = np.asarray(solver.get_joint_state(), dtype=float)
        T = np.asarray(solver.current_pose(), dtype=float)
        self.set_joint_values(q)
        self.update_robot_from_joint_values(q)
        self.update_ee_display(T)
        if move_gizmo:
            self.set_target_pose(T)