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

"""robokin — backend-agnostic robot kinematics helpers."""

from robokin.robot_model import RobotModel, load_robot_description

__all__ = [
    "RobotModel",
    "load_robot_description",
    "PlacoKinematics",
    "PlacoConfig",
    "PyrokiKinematics",
    "PyrokiConfig",
    "RoboPlanOinkKinematics",
    "OinkConfig",
    "ViserRobotUI",
    "RerunRobotLogger",
]

# ── Optional backend re-exports (import only when the dep is installed) ──

def __getattr__(name: str):
    """Lazy-load backend classes so missing deps don't break the package."""
    _lazy = {
        "PlacoKinematics": ("robokin.placo", "PlacoKinematics"),
        "PlacoConfig": ("robokin.placo", "PlacoConfig"),
        "PyrokiKinematics": ("robokin.pyroki", "PyrokiKinematics"),
        "PyrokiConfig": ("robokin.pyroki", "PyrokiConfig"),
        "RoboPlanOinkKinematics": ("robokin.roboplan_oink", "RoboPlanOinkKinematics"),
        "OinkConfig": ("robokin.roboplan_oink", "OinkConfig"),
        "ViserRobotUI": ("robokin.ui.viser_app", "ViserRobotUI"),
        "RerunRobotLogger": ("robokin.ui.rerun_logger", "RerunRobotLogger"),
    }
    if name in _lazy:
        import importlib
        mod_path, attr = _lazy[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'robokin' has no attribute {name!r}")