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

"""robokin.ui.rerun_logger — Rerun visualization helpers for robot kinematics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import rerun as rr
    import rerun.blueprint as rrb
    from rerun.urdf import UrdfTree

    _HAS_RERUN = True
except ImportError:  # pragma: no cover
    _HAS_RERUN = False


# ---------------------------------------------------------------------------
# Rerun session manager
# ---------------------------------------------------------------------------

class RerunRobotLogger:
    """Thin helper that logs URDF transforms, joint angles, and EE position
    to a Rerun session.

    Usage::

        logger = RerunRobotLogger(
            urdf_path="robot.urdf",
            joint_names=["j1", "j2", "j3"],
            ee_frame="tool0",
            app_id="my_app",
        )
        logger.init()

        # in your control loop (radians):
        logger.log_state(joint_rad, ee_T)

        # or degrees:
        logger.log_state_deg(joint_deg, ee_T)
    """

    def __init__(
        self,
        urdf_path: str | Path,
        joint_names: list[str],
        ee_frame: str,
        app_id: str = "robokin",
    ) -> None:
        if not _HAS_RERUN:
            raise ImportError("rerun-sdk is required: pip install rerun-sdk")

        self.urdf_path = str(urdf_path)
        self.joint_names = list(joint_names)
        self.ee_frame = ee_frame
        self.app_id = app_id

        self._urdf_tree: UrdfTree | None = None
        self._step = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init(self, *, serve_web: bool = True) -> None:
        """Initialize Rerun, load URDF, log static axes, send blueprint."""
        rr.init(self.app_id)
        server_uri = rr.serve_grpc()
        if serve_web:
            rr.serve_web_viewer(connect_to=server_uri)

        rr.log_file_from_path(self.urdf_path, static=True)
        self._urdf_tree = UrdfTree.from_file_path(self.urdf_path)
        self._log_axes_static()
        self._send_default_blueprint()

    # ------------------------------------------------------------------
    # Static setup
    # ------------------------------------------------------------------

    def _log_axes_static(self) -> None:
        if self._urdf_tree is None:
            return
        link_names: set[str] = set()
        for joint in self._urdf_tree.joints():
            link_names.add(joint.parent_link)
            link_names.add(joint.child_link)
        for link_name in sorted(link_names):
            rr.log(
                f"axes/{link_name}",
                rr.Transform3D(parent_frame=link_name),
                rr.TransformAxes3D(0.06),
                static=True,
            )
            is_ee = link_name == self.ee_frame
            rr.log(
                f"axes/{link_name}/label",
                rr.Points3D(
                    [[0, 0, 0]],
                    radii=0.01 if is_ee else 0.004,
                    colors=[[0, 255, 0]] if is_ee else [[255, 255, 255]],
                    labels=[link_name],
                ),
                static=True,
            )

    def _send_default_blueprint(self) -> None:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Robot", origin="/"),
                rrb.Vertical(
                    rrb.TimeSeriesView(name="Joint Angles (rad)", origin="joints"),
                    rrb.TimeSeriesView(name="EE Position", origin="ee"),
                ),
                column_shares=[2, 1],
            ),
            auto_layout=False,
            auto_views=False,
        )
        rr.send_blueprint(blueprint)

    # ------------------------------------------------------------------
    # Per-tick logging
    # ------------------------------------------------------------------

    def log_state(
        self,
        joint_rad: np.ndarray,
        ee_T: np.ndarray,
        label: str = "",
    ) -> None:
        """Log joint angles (radians), URDF transforms, and EE position."""
        self._step += 1
        rr.set_time("step", sequence=self._step)

        q_rad = np.asarray(joint_rad, dtype=float)
        angle_map = {name: float(q_rad[i]) for i, name in enumerate(self.joint_names)}

        if self._urdf_tree is not None:
            for joint in self._urdf_tree.joints():
                if joint.joint_type == "revolute":
                    rr.log(
                        "transforms",
                        joint.compute_transform(
                            angle_map.get(joint.name, 0.0), clamp=True
                        ),
                    )

        for i, name in enumerate(self.joint_names):
            rr.log(f"joints/{name}", rr.Scalars(float(q_rad[i])))

        rr.log("ee/x_mm", rr.Scalars(float(ee_T[0, 3] * 1000.0)))
        rr.log("ee/y_mm", rr.Scalars(float(ee_T[1, 3] * 1000.0)))
        rr.log("ee/z_mm", rr.Scalars(float(ee_T[2, 3] * 1000.0)))

    def log_state_deg(
        self,
        joint_deg: np.ndarray,
        ee_T: np.ndarray,
        label: str = "",
    ) -> None:
        """Convenience wrapper: accepts joint angles in degrees."""
        self.log_state(np.deg2rad(joint_deg), ee_T, label=label)

