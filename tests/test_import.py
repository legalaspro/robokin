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

"""Smoke tests — make sure the core robokin package imports cleanly."""

import importlib


def test_import_robokin():
    """robokin top-level imports without error."""
    import robokin
    assert hasattr(robokin, "RobotModel")
    assert hasattr(robokin, "load_robot_description")


def test_import_transformations():
    """robokin.transformations imports without error."""
    from robokin.transformations import (
        ease_in_out_sine,
        interpolate_pose,
        pose_distance,
        compute_segment_steps_from_speed,
    )
    assert callable(ease_in_out_sine)
    assert callable(interpolate_pose)
    assert callable(pose_distance)
    assert callable(compute_segment_steps_from_speed)


def test_import_robot_model():
    from robokin.robot_model import RobotModel
    assert RobotModel is not None


def test_lazy_attr_error():
    """Accessing a non-existent attribute raises AttributeError."""
    import robokin
    import pytest
    with pytest.raises(AttributeError):
        _ = robokin.NoSuchThing