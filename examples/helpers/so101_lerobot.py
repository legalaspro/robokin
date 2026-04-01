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

import threading
import time
from dataclasses import dataclass

import numpy as np
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig


ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
GRIPPER_NAME = "gripper"
ALL_MOTOR_NAMES = ARM_JOINT_NAMES + [GRIPPER_NAME]


@dataclass
class SO101LeRobotConfig:
    port: str = "/dev/so101_follower"
    calibration_id: str = "lerobot_follower_arm"
    use_degrees: bool = True


class SO101LeRobotArm:
    def __init__(
        self,
        cfg: SO101LeRobotConfig | None = None,
        solver_joint_names: list[str] | None = None,
    ):
        self.cfg = cfg or SO101LeRobotConfig()
        self.robot: SO100Follower | None = None
        self.gripper: float = 0.0
        self._lock = threading.Lock()

        # Hardware motor order (fixed)
        self._hw_joint_names = list(ALL_MOTOR_NAMES)

        # Solver order — the order q vectors arrive in from the solver.
        # If not provided, assume it matches hardware order.
        self._solver_joint_names = solver_joint_names or list(self._hw_joint_names)

        # Build reorder maps: solver index → hw index and vice versa
        self._solver_to_hw = [
            self._hw_joint_names.index(n) for n in self._solver_joint_names
        ]
        self._hw_to_solver = [
            self._solver_joint_names.index(n) for n in self._hw_joint_names
        ]

    def connect(self, verify_motor_order: bool = True) -> None:
        follower_cfg = SO100FollowerConfig(
            port=self.cfg.port,
            id=self.cfg.calibration_id,
            use_degrees=self.cfg.use_degrees,
        )
        self.robot = SO100Follower(follower_cfg)
        self.robot.connect()

        if verify_motor_order:
            motor_names = list(self.robot.bus.motors.keys())
            if motor_names != ALL_MOTOR_NAMES:
                raise RuntimeError(
                    f"Unexpected motor order: {motor_names} != {ALL_MOTOR_NAMES}"
                )

        # Cache initial gripper position from hardware
        obs = self.robot.get_observation()
        self.gripper = float(obs[f"{GRIPPER_NAME}.pos"])

    def disconnect(self) -> None:
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None

    def _read_observation(self, retries: int = 3, delay: float = 0.05) -> dict:
        """Read observation with retries to handle transient serial errors."""
        for attempt in range(retries):
            try:
                with self._lock:
                    return self.robot.get_observation()
            except ConnectionError:
                if attempt == retries - 1:
                    raise
                time.sleep(delay)

    def get_observation(self) -> dict:
        return self._read_observation()

    def get_joint_state(self) -> np.ndarray:
        """Read all 6 joint positions from hardware in **solver** order. Returns radians."""
        obs = self._read_observation()
        # Read in hardware order
        hw_deg = np.array([obs[f"{name}.pos"] for name in self._hw_joint_names], dtype=float)
        hw_rad = np.deg2rad(hw_deg)
        # Reorder to solver order
        return hw_rad[self._solver_to_hw]

    def set_joint_state(self, q: np.ndarray) -> None:
        """Send all 6 joint positions in **solver** order (radians).

        Gripper is always overwritten with the cached value (degrees).
        """
        q_rad = np.asarray(q, dtype=float)
        # Reorder from solver order to hardware order
        hw_rad = q_rad[self._hw_to_solver]
        hw_deg = np.rad2deg(hw_rad)
        action = {f"{name}.pos": float(hw_deg[i]) for i, name in enumerate(self._hw_joint_names)}
        action[f"{GRIPPER_NAME}.pos"] = self.gripper
        with self._lock:
            self.robot.send_action(action)

    def set_gripper_position(self, value: float) -> None:
        """Update cached gripper value (radians) and immediately send only the gripper."""
        self.gripper = float(np.rad2deg(value))
        with self._lock:
            self.robot.send_action({f"{GRIPPER_NAME}.pos": self.gripper})

    def set_pid_all(self, p: int, i: int, d: int) -> None:
        bus = self.robot.bus
        with bus.torque_disabled(ARM_JOINT_NAMES):
            for motor_name in ARM_JOINT_NAMES:
                bus.write("P_Coefficient", motor_name, p, normalize=False)
                bus.write("I_Coefficient", motor_name, i, normalize=False)
                bus.write("D_Coefficient", motor_name, d, normalize=False)