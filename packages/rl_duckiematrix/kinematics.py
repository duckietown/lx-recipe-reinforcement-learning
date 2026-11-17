"""From the policy's steering to the wheels of a real Duckiebot.

In training, a steering value becomes two wheel PWMs (:func:`~rl_duckiematrix.env.steering_to_wheels`
and the gym environment's PWM scale), and the simulator turns those into a forward speed ``v`` and a
turn rate ``omega``. A real robot's motors answer the same PWM differently, and each robot has its own
trim. So on a real robot the agent asks for the ``v`` and ``omega`` the simulator produced, and turns
them into PWMs with the robot's kinematics calibration, exactly like the Duckiebot's kinematics node
does for ``car_cmd``. A virtual robot is the simulator itself, so it keeps the training PWMs.

The calibration is ``/data/config/calibrations/kinematics/<robot>.yaml`` on the robot, or
``default.yaml`` there if the robot was never calibrated. When the agent runs on the laptop
(``dts code workbench --local``), it reads the same files through the robot's files API.
"""

from __future__ import annotations

import dataclasses
import urllib.request
from pathlib import Path

import numpy as np
import yaml

from rl_duckiematrix.env import WHEEL_COMMAND_TO_PWM, steering_to_wheels

# The Duckiematrix's response to PWM, measured on the recipe's map with delta_t 0.1 (constant wheel
# commands, 3 s each, steering -1 to 1): exactly linear, 0.144 m/s and up to 1.20 rad/s with the
# default steering mapping. A positive turn rate turns left (right wheel faster).
SIM_SPEED_PER_PWM = 0.600  # m/s per unit of mean PWM
SIM_TURN_RATE_PER_PWM = 3.74  # rad/s per unit of (right PWM - left PWM)

DATA_DIR = Path("/data")
CALIBRATION_DIR = "config/calibrations/kinematics"
ROBOT_HARDWARE_FILE = "config/robot_hardware"
LOCAL_HOSTS = ("127.0.0.1", "localhost")


@dataclasses.dataclass
class KinematicsCalibration:
    """The fields of a Duckiebot kinematics calibration file, with the kinematics node's defaults."""

    gain: float = 1.0
    trim: float = 0.0
    k: float = 27.0
    radius: float = 0.0318
    baseline: float = 0.1
    limit: float = 1.0
    v_max: float = 1.0
    omega_max: float = 8.0
    source: str = "built-in defaults"

    @classmethod
    def from_yaml(cls, text: str, source: str) -> KinematicsCalibration:
        values = yaml.safe_load(text) or {}
        fields = {f.name for f in dataclasses.fields(cls)} - {"source"}
        return cls(**{name: float(values[name]) for name in fields if values.get(name) is not None}, source=source)

    def wheels(self, v: float, omega: float) -> tuple[float, float]:
        """``(left, right)`` PWM for a forward speed ``v`` (m/s) and turn rate ``omega`` (rad/s, positive = left).

        Same computation as the Duckiebot's kinematics node: inverse kinematics of the differential
        drive, then the motor constant ``k`` corrected by ``gain`` and ``trim``.
        """
        v = float(np.clip(v, -self.v_max, self.v_max))
        omega = float(np.clip(omega, -self.omega_max, self.omega_max))
        omega_right = (v + 0.5 * omega * self.baseline) / self.radius
        omega_left = (v - 0.5 * omega * self.baseline) / self.radius
        right = omega_right * (self.gain + self.trim) / self.k
        left = omega_left * (self.gain - self.trim) / self.k
        return float(np.clip(left, -self.limit, self.limit)), float(np.clip(right, -self.limit, self.limit))


def steering_to_velocity(steering: float, speed: float, gain: float) -> tuple[float, float]:
    """The ``(v, omega)`` the simulator produced for ``steering`` with the policy's steering mapping."""
    left, right = WHEEL_COMMAND_TO_PWM * steering_to_wheels(steering, speed, gain)
    return SIM_SPEED_PER_PWM * 0.5 * (left + right), SIM_TURN_RATE_PER_PWM * (right - left)


def read_robot_file(host: str, path: str, timeout: float = 3.0) -> str | None:
    """A file of the robot's ``/data`` directory: from disk on the robot, else from its files API."""
    if host in LOCAL_HOSTS:
        local = DATA_DIR / path
        return local.read_text() if local.is_file() else None
    try:
        with urllib.request.urlopen(f"http://{host}/files/data/{path}", timeout=timeout) as response:
            return response.read().decode()
    except OSError:
        return None


def is_virtual_robot(host: str) -> bool:
    text = read_robot_file(host, ROBOT_HARDWARE_FILE)
    return text is not None and text.strip() == "virtual"


def load_kinematics_calibration(robot_name: str, host: str) -> KinematicsCalibration:
    """The robot's kinematics calibration (``<robot>.yaml``, else ``default.yaml``, else the defaults)."""
    for name in (robot_name, "default"):
        path = f"{CALIBRATION_DIR}/{name}.yaml"
        text = read_robot_file(host, path)
        if text is not None:
            return KinematicsCalibration.from_yaml(text, source=f"/data/{path} on {robot_name}")
    return KinematicsCalibration()
