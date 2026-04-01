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

"""Integration test for PlacoKinematics — requires placo to be installed."""

from pathlib import Path

import numpy as np
import pytest

try:
    import placo  # noqa: F401
    _HAS_PLACO = True
except ImportError:
    _HAS_PLACO = False


pytestmark = pytest.mark.skipif(not _HAS_PLACO, reason="placo not installed")


@pytest.fixture
def urdf_path():
    """Return path to SO-101 URDF if available, else skip."""
    # Try robot_descriptions first
    try:
        from robokin.robot_model import load_robot_description
        model = load_robot_description("so_arm101_description")
        return str(model.urdf_path)
    except Exception:
        pass
    pytest.skip("No SO-101 URDF found")


EE_FRAME = "gripper_frame_link"

def test_placo_fk(urdf_path):
    from robokin.placo import PlacoKinematics, PlacoConfig

    solver = PlacoKinematics(
        urdf_path=urdf_path,
        ee_frame=EE_FRAME,
        cfg=PlacoConfig(),
    )
    T = solver.fk(np.zeros(len(solver.joint_names)))
    assert T.shape == (4, 4)
    # EE should be above ground at zero config
    assert T[2, 3] > 0.0


def test_placo_roundtrip_deg(urdf_path):
    from robokin.placo import PlacoKinematics

    solver = PlacoKinematics(
        urdf_path=urdf_path,
        ee_frame=EE_FRAME,
    )
    n = len(solver.joint_names)
    rng = np.random.default_rng(42)
    q_deg = rng.uniform(-20.0, 20.0, size=n)
    solver.set_joint_state_deg(q_deg)
    q_back = solver.get_joint_state_deg()
    np.testing.assert_allclose(q_back, q_deg, atol=0.1)