"""The Duckiematrix environment in real-time mode (notebook 06).

In gym mode (:class:`rl_duckiematrix.env.DuckiematrixEnv`) the engine waits for the agent: each step applies
the action for exactly ``delta_t`` of simulated time and returns the image rendered at that instant, and the
agent's inference and gradient updates cost no simulated time. In real-time mode, the Duckiematrix's normal
mode (virtual robots, the workbench), the engine runs on the wall clock. :class:`RealtimeDuckiematrixEnv`
sends the wheel commands over DTPS, as the robot agent does, and reads the newest camera image and pose: the
robot keeps executing the last command while the agent thinks, and a slow decision arrives late.

Everything else is the same as in gym mode: observations, actions, reward and ``info["lane"]`` (the
computation is ``DuckiematrixEnv``'s), and :func:`rl_duckiematrix.env.make_env` wraps it the same way
(``mode="realtime"``), random start poses included. Start the engine without ``--gym``
(``dts code start_matrix``).
"""

from __future__ import annotations

import math
import time
from typing import Any

import gymnasium as gym
import numpy as np
from duckietown.sdk.robots.duckiebot import DB21J
from duckietown.sdk.utils.lane_position import LanePositionCalculator, MapInterpreter
from gymnasium import spaces

from rl_duckiematrix import env as gym_env
from rl_duckiematrix.engine import resolve_engine_host

# a pose reset is done when the robot reports a position this close to the requested one (m)
_POSE_REACHED = 0.01


class RealtimeDuckiematrixEnv(gym.Env):
    """``DuckiematrixEnv``'s observations, actions, reward and ``info["lane"]``, on the wall clock.

    ``step(action)`` sends the wheel commands, waits until the next tick (``tick`` seconds after the previous
    one: 0.1 s, the gym mode's ``delta_t``), then returns the newest camera image and the reward for the pose
    change since the previous step. If the caller comes back late (a long gradient update), there is no wait
    and the lateness is recorded in :attr:`timing`. ``reset()`` puts the robot back at its spawn pose;
    ``reset(options={"pose": {"x": ..., "y": ..., "yaw": ...}})`` at that pose (what ``RandomStart`` does).

    Args:
        engine_host: engine address; resolved with :func:`resolve_engine_host` when ``None``.
        tick: seconds between two steps.
        reward_fn: as in ``DuckiematrixEnv``.
        out_of_road_penalty: reward when the robot leaves the road (ends the episode).
        timeout: seconds to wait for the map, the first image and pose resets.

    """

    metadata = {"render_modes": []}  # noqa: RUF012

    # the lane geometry, reward and helpers are DuckiematrixEnv's: they only need the attributes set below
    _get_reward = gym_env.DuckiematrixEnv._get_reward  # noqa: SLF001
    heading = gym_env.DuckiematrixEnv.heading
    tile_kind = gym_env.DuckiematrixEnv.tile_kind

    def __init__(
        self,
        engine_host: str | None = None,
        tick: float = gym_env.DELTA_T,
        reward_fn: gym_env.RewardFn | None = None,
        out_of_road_penalty: float = -10.0,
        timeout: float = 20.0,
    ) -> None:
        self._engine_host = engine_host or resolve_engine_host()
        self._tick = float(tick)
        self._delta_t = float(tick)  # nominal; the reward uses the measured interval (see _lane_step)
        self._reward_fn = reward_fn or gym_env.default_reward
        self._out_of_road_penalty = out_of_road_penalty
        self._timeout = timeout
        self._lane_info: dict[str, float] = {}
        self._map_pose: tuple[np.ndarray, float] | None = None
        self._observation: np.ndarray | None = None
        self._pose: dict | None = None
        self._previous_pose: dict | None = None
        self._next_tick: float | None = None
        self.timing: dict[str, list[float]] = {"late_s": [], "pose_age_s": []}
        self.robot = DB21J(gym_env.ENTITY_NAME, host=self._engine_host, simulated=True)
        self._camera = self.robot.camera
        self._motors = self.robot.motors
        # only what this environment uses (DB21J.start() also wants an IMU, which the recipe's map lacks); the
        # map layers start on their first get()
        self._components = [self.robot.pose, self.robot.state_reset_flag, self._camera, self._motors]
        try:
            for component in self._components:
                component.start()
            layers = {}
            for name, component in (("frames", self.robot.map_frames), ("tiles", self.robot.map_tiles),
                                    ("tile_info", self.robot.map_tile_info)):
                layer = component.get(block=True, clean_up=True, timeout=timeout)
                if layer is None:
                    message = (f"No map from the engine at {self._engine_host}: is the Duckiematrix running, "
                               "in real-time mode (without --gym)?")
                    raise TimeoutError(message)  # noqa: TRY301
                layers[name] = layer
        except BaseException:
            self.close()
            raise
        self._map_interpreter = MapInterpreter(layers)
        self._lane_position_calculator = LanePositionCalculator(self._map_interpreter)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = spaces.Box(
            0, 255, (gym_env.CAMERA_HEIGHT, gym_env.CAMERA_WIDTH, 3), np.uint8,
        )

    @property
    def engine_host(self) -> str:
        return self._engine_host

    # -- sensors

    def _latest(self, *, wait_for_new: bool = False) -> tuple[np.ndarray, dict]:
        """The newest camera image (RGB, as in gym mode) and pose."""
        deadline = time.monotonic() + self._timeout
        previous = self._pose["timestamp"] if (wait_for_new and self._pose) else None
        while True:
            image, pose = self._camera.latest, self.robot.pose.latest
            if image is not None and pose is not None and (previous is None or pose["timestamp"] > previous):
                break
            if time.monotonic() > deadline:
                message = "No camera image or pose from the engine: is it running in real-time mode?"
                raise TimeoutError(message)
            time.sleep(0.005)
        # the SDK decodes to BGR; gym mode returns RGB
        return np.ascontiguousarray(image[:, :, ::-1]), pose

    def _lane_step(self, pose: dict, previous_pose: dict) -> tuple[float, bool]:
        # speeds over the time that really passed, so a late step is not rewarded for the extra distance
        interval = pose["timestamp"] - previous_pose["timestamp"]
        self._delta_t = interval if interval > 1e-3 else self._tick  # noqa: PLR2004
        try:
            return self._get_reward(pose, previous_pose)
        finally:
            self._delta_t = self._tick

    # -- gym API

    def step(self, action):
        values = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        pwm = gym_env.WHEEL_COMMAND_TO_PWM * values
        self._motors.set(float(pwm[0]), float(pwm[1]))
        now = time.monotonic()
        if self._next_tick is None:
            self._next_tick = now + self._tick
        late = now - self._next_tick
        if late < 0:
            time.sleep(-late)
        self.timing["late_s"].append(max(0.0, late))
        self._next_tick = max(self._next_tick + self._tick, time.monotonic())
        self._observation, self._pose = self._latest()
        self.timing["pose_age_s"].append(max(0.0, time.time() - self._pose["timestamp"]))
        reward, terminated = self._lane_step(self._pose, self._previous_pose or self._pose)
        self._previous_pose = self._pose
        if terminated:
            self._motors.set(0.0, 0.0)
        return self._observation, reward, terminated, False, {"lane": dict(self._lane_info)}

    def _reset_to_pose(self, pose: dict) -> None:
        from duckietown_messages.geometry_3d import Position, Quaternion, Transformation  # noqa: PLC0415

        component = self.robot.state_reset_pose
        if not component.has_started:
            component.start()
            self._components.append(component)
        half_yaw = 0.5 * float(pose["yaw"])
        component.set(Transformation(
            position=Position(x=float(pose["x"]), y=float(pose["y"]), z=float(pose.get("z", 0.0))),
            rotation=Quaternion(w=math.cos(half_yaw), x=0.0, y=0.0, z=math.sin(half_yaw)),
        ))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            reached = self.robot.pose.latest
            if reached and math.hypot(reached["position"]["x"] - pose["x"],
                                      reached["position"]["y"] - pose["y"]) < _POSE_REACHED:
                return
            time.sleep(0.01)
        message = (
            f"The robot did not reach the reset pose {pose}: this Duckiematrix engine cannot reset to a pose in "
            "real time (it needs DTSW-8541); update it, or use make_env(..., random_start=False)."
        )
        raise RuntimeError(message)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._motors.set(0.0, 0.0)
        if options and options.get("pose") is not None:
            self._reset_to_pose(options["pose"])
        else:
            self.robot.state_reset_flag.set(data=True)  # back to the spawn pose
        time.sleep(0.3)  # let the engine apply the reset and a new image through
        self._observation, self._pose = self._latest(wait_for_new=True)
        self._previous_pose = self._pose
        self._get_reward(self._pose, self._pose)  # lane information at the new pose
        self._next_tick = None
        return self._observation, {"lane": dict(self._lane_info)}

    def latest_image(self) -> np.ndarray | None:
        """The last camera image (RGB, uint8)."""
        return self._observation

    def close(self) -> None:
        try:
            if self._motors.has_started:
                self._motors.set(0.0, 0.0)
        except Exception:  # noqa: BLE001, S110
            pass
        for component in reversed(self._components):
            try:
                if component.has_started:
                    component.stop()
            except Exception:  # noqa: BLE001, S110
                pass


def timing_summary(env: gym.Env, *, clear: bool = False) -> dict[str, Any]:
    """How late the real-time steps were (empty in gym mode); ``clear`` starts a new count."""
    timing = getattr(env.unwrapped, "timing", None)
    if not timing or not timing["late_s"]:
        return {}
    late = np.array(timing["late_s"])
    age = np.array(timing["pose_age_s"])
    is_late = late > 0.01  # noqa: PLR2004
    summary = {
        "steps": len(late),
        "late_steps_pct": round(100 * float(is_late.mean()), 1),
        "late_ms_when_late": round(1000 * float(late[is_late].mean()), 1) if is_late.any() else 0.0,
        "late_max_ms": round(1000 * float(late.max()), 1),
        "pose_age_ms": round(1000 * float(age.mean()), 1),
    }
    if clear:
        for values in timing.values():
            values.clear()
    return summary
