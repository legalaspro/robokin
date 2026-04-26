# Copyright 2026 Sebastian Castro
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

"""Integration test for RoboPlanOinkKinematics — requires RoboPlan to be installed."""

import numpy as np
import pytest

try:
    import roboplan  # noqa: F401
    _HAS_ROBOPLAN = True
except ImportError:
    _HAS_ROBOPLAN = False


pytestmark = pytest.mark.skipif(not _HAS_ROBOPLAN, reason="roboplan not installed")


@pytest.fixture
def model_paths():
    """Return path to SO-101 URDF and SRDF, else skip."""
    try:
        from roboplan.example_models import get_package_models_dir
        so101_dir = get_package_models_dir() / "so101_robot_model"
        urdf_path = so101_dir / "so101.urdf"
        srdf_path = so101_dir / "so101.srdf"
        return (urdf_path, srdf_path)
    except Exception:
        pass
    pytest.skip("No SO-101 RoboPlan models found")


EE_FRAME = "gripper_frame_link"

def test_roboplan_oink_fk(model_paths):
    from robokin.roboplan_oink import RoboPlanOinkKinematics, OinkConfig
    from roboplan.example_models import get_package_share_dir

    urdf_path, srdf_path = model_paths
    solver = RoboPlanOinkKinematics(
        urdf_path=urdf_path,
        srdf_path=srdf_path,
        package_paths=[get_package_share_dir()],
        ee_frame=EE_FRAME,
        cfg=OinkConfig(),
    )
    T = solver.fk(np.zeros(len(solver.group_info.joint_names)))
    assert T.shape == (4, 4)
    # EE should be above ground at zero config
    assert T[2, 3] > 0.0


def test_roboplan_oink_roundtrip_deg(model_paths):
    from robokin.roboplan_oink import RoboPlanOinkKinematics
    from roboplan.example_models import get_package_share_dir

    urdf_path, srdf_path = model_paths
    solver = RoboPlanOinkKinematics(
        urdf_path=urdf_path,
        srdf_path=srdf_path,
        package_paths=[get_package_share_dir()],
        ee_frame=EE_FRAME,
    )
    rng = np.random.default_rng(42)
    q_deg = rng.uniform(-20.0, 20.0, size=len(solver.group_info.joint_names))
    solver.set_joint_state_deg(q_deg)
    q_back = solver.get_joint_state_deg()
    np.testing.assert_allclose(q_back, q_deg, atol=0.1)
