"""Drive a Duckiebot (real, or virtual in the Duckiematrix) with the trained policy.

Started by the launcher through ``dts code workbench -R ROBOT`` (add ``-m`` for the Duckiematrix).
The loop is the deployment counterpart of the training environment: camera image -> the learner's
``preprocess`` -> frame stack -> greedy action -> wheel commands, at the rate the agent acted at in
training (the policy's ``decision_interval``: the engine's ``delta_t`` times the action repeat, 0.2 s
of simulated time with the recipe's defaults, i.e. 5 Hz).

Environment variables:
    VEHICLE_NAME: the robot (set by ``dts``).
    ROBOT_HOST: DTPS host of the robot (default: ``127.0.0.1`` when the robot's DTPS port answers
        there, i.e. when the agent runs on the robot, otherwise ``<VEHICLE_NAME>.local``).
    POLICY_PATH: the ONNX file (default ``<project>/assets/models/policy.onnx``).
    AGENT_RATE_HZ: control rate (default: ``1 / decision_interval`` of the policy).
    AGENT_DURATION_S: stop after this many seconds (default: run until stopped).
"""

from __future__ import annotations

import os
import signal
import socket
import time
from pathlib import Path

from duckietown.sdk.robots.duckiebot import DB21J

from rl_duckiematrix.env import WHEEL_COMMAND_TO_PWM
from rl_duckiematrix.export import OnnxPolicy
from solution.preprocessing import preprocess

PROJECT_PATH = Path(os.environ.get("DT_REPO_PATH", Path(__file__).resolve().parents[2]))
DEFAULT_POLICY = PROJECT_PATH / "assets" / "models" / "policy.onnx"
ROBOT_DTPS_PORT = 11911


def robot_host(name: str) -> str:
    """``ROBOT_HOST``, else the local DTPS port (agent running on the robot), else ``<name>.local``."""
    explicit = os.environ.get("ROBOT_HOST")
    if explicit:
        return explicit
    try:
        with socket.create_connection(("127.0.0.1", ROBOT_DTPS_PORT), timeout=1.0):
            return "127.0.0.1"
    except OSError:
        return f"{name}.local"


def main() -> None:  # noqa: PLR0915
    name = os.environ.get("VEHICLE_NAME")
    if not name:
        message = "VEHICLE_NAME is not set; run this through `dts code workbench -R ROBOT`."
        raise SystemExit(message)
    policy_path = Path(os.environ.get("POLICY_PATH", DEFAULT_POLICY))
    if not policy_path.exists():
        message = (
            f"No policy at {policy_path}. Export it from the training notebook to assets/models/ and "
            "rebuild with `dts code build -R ROBOT`."
        )
        raise SystemExit(message)
    duration = float(os.environ.get("AGENT_DURATION_S", "0"))
    host = robot_host(name)

    policy = OnnxPolicy(policy_path)
    print(f"[agent] policy {policy_path.name}: {policy.meta}")
    rate = float(os.environ.get("AGENT_RATE_HZ", 1.0 / policy.decision_interval))
    robot = DB21J(name, host=host)
    camera = robot.camera
    motors = robot.motors
    camera.start()
    motors.start()
    print(f"[agent] connected to {name} at {host}; driving at {rate:.0f} Hz. Ctrl+C to stop.")

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    policy.reset()
    period = 1.0 / rate
    started = time.monotonic()
    steps = 0
    try:
        while running:
            tick = time.monotonic()
            if duration and tick - started > duration:
                break
            image = camera.get(block=True, timeout=2.0)
            if image is None:
                print("[agent] no camera image for 2 s, stopping the wheels")
                motors.set(0.0, 0.0)
                continue
            # the SDK decodes to BGR; training images are RGB
            frame = preprocess(image[:, :, ::-1])
            action = policy.act(frame)
            left, right = (WHEEL_COMMAND_TO_PWM * float(v) for v in policy.wheels(action))
            motors.set(left, right)
            steps += 1
            if steps % max(int(rate), 1) == 0:
                print(f"[agent] step {steps}: steering {policy.steering(action):+.2f} -> PWM ({left:+.2f}, {right:+.2f})")
            time.sleep(max(0.0, period - (time.monotonic() - tick)))
    finally:
        motors.set(0.0, 0.0)
        motors.stop()
        camera.stop()
        print(f"[agent] stopped after {steps} steps")


if __name__ == "__main__":
    main()
