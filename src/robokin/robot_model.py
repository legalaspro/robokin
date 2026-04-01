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

import os
from dataclasses import dataclass
from pathlib import Path
import importlib

# Pinned SO-ARM100 upstream commit with semantic joint/link names.
_SO101_DEFAULT_COMMIT = "385e8d7c68e24945df6c60d9bd68837a4b7411ae"


@dataclass(slots=True)
class RobotModel:
    name: str
    urdf_path: Path
    package_path: Path


def load_robot_description(name: str = "so_arm101_description") -> RobotModel:
    """Load a robot description via ``robot_descriptions``.

    For *so_arm101_description* the SO-ARM100 repo is pinned to
    ``_SO101_DEFAULT_COMMIT`` (override with ``ROBOT_DESCRIPTION_COMMIT``).
    """
    if name == "so_arm101_description":
        os.environ.setdefault("ROBOT_DESCRIPTION_COMMIT", _SO101_DEFAULT_COMMIT)

    mod = importlib.import_module(f"robot_descriptions.{name}")

    return RobotModel(
        name=name,
        urdf_path=Path(mod.URDF_PATH),
        package_path=Path(mod.PACKAGE_PATH),
    )