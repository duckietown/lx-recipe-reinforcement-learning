"""Export a trained policy to ONNX and run it without PyTorch.

Two kinds of policy are supported:

- ``"discrete"``: a Q-network (DQN). The ONNX model returns the Q-values of the five steering
  actions; the policy takes the action with the largest one.
- ``"steering"``: a continuous policy (SAC). The ONNX model returns one steering value in [-1, 1].

Either way the exported file takes the stacked, preprocessed observation (``uint8``, channel-first,
with a batch dimension). A JSON sidecar stores what the robot needs to reproduce the training-time
observation and action: the frame-stack depth, the decision interval and the steering table. The
image preprocessing itself is the learner's ``solution.preprocessing.preprocess``, which runs both in
training and on the robot.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from rl_duckiematrix.env import STEERING_GAIN, STEERING_SPEED, STEERING_VALUES, steering_to_wheels

ONNX_INPUT = "observation"
ONNX_OUTPUTS = {"discrete": "q_values", "steering": "steering"}


def export_policy(  # noqa: PLR0913
    model,
    observation_shape: tuple[int, ...],
    path: str | Path,
    *,
    frame_stack: int,
    decision_interval: float,
    action_type: str = "discrete",
    speed: float = STEERING_SPEED,
    gain: float = STEERING_GAIN,
) -> Path:
    """Write ``path`` (``.onnx``) and ``path.with_suffix('.json')``; returns the ONNX path.

    Args:
        model: a ``torch.nn.Module`` mapping stacked ``uint8`` frames to Q-values
            (``action_type="discrete"``) or to a steering value (``action_type="steering"``).
        observation_shape: the shape of one stacked observation.
        path: the ``.onnx`` file to write.
        frame_stack: how many frames the observation stacks.
        decision_interval: simulated seconds between two decisions in training (``delta_t`` times
            the action repeat); the robot agent acts at the same rate.
        action_type: ``"discrete"`` or ``"steering"``.
        speed, gain: the steering-to-wheels mapping used in training.

    """
    import torch  # noqa: PLC0415

    if action_type not in ONNX_OUTPUTS:
        message = f"action_type must be one of {list(ONNX_OUTPUTS)}, got {action_type!r}."
        raise ValueError(message)
    output = ONNX_OUTPUTS[action_type]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.zeros((1, *observation_shape), dtype=torch.uint8)
    torch.onnx.export(
        model, dummy, str(path), input_names=[ONNX_INPUT], output_names=[output],
        dynamic_axes={ONNX_INPUT: {0: "batch"}, output: {0: "batch"}}, opset_version=17,
        dynamo=False,
    )
    meta = {
        "action_type": action_type,
        "observation_shape": list(observation_shape),
        "frame_stack": frame_stack,
        "decision_interval": decision_interval,
        "steering_values": list(STEERING_VALUES),
        "speed": speed,
        "gain": gain,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


class OnnxPolicy:
    """Deterministic policy from an exported ONNX model, with its own frame stack.

    Call :meth:`reset` at the start of an episode, then :meth:`act` with each preprocessed frame
    (``uint8``, ``(H, W)`` or ``(H, W, C)``, as returned by ``preprocess``). Actions are action
    indices for a discrete policy and steering values for a continuous one; :meth:`wheels` turns
    either into wheel commands.
    """

    def __init__(self, path: str | Path) -> None:
        import onnxruntime as ort  # noqa: PLC0415

        path = Path(path)
        self.meta = json.loads(path.with_suffix(".json").read_text())
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._frames: deque[np.ndarray] = deque(maxlen=int(self.meta["frame_stack"]))
        self.last_output: np.ndarray | None = None

    @property
    def action_type(self) -> str:
        return self.meta.get("action_type", "discrete")

    @property
    def frame_stack(self) -> int:
        return int(self.meta["frame_stack"])

    @property
    def decision_interval(self) -> float:
        """Seconds between two decisions in training (the robot agent acts at the same rate)."""
        return float(self.meta.get("decision_interval", 0.2))

    @property
    def last_q_values(self) -> np.ndarray | None:
        """The Q-values of the last decision (discrete policies)."""
        return self.last_output if self.action_type == "discrete" else None

    def reset(self) -> None:
        self._frames.clear()

    @staticmethod
    def _channel_first(frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim == 2:
            return frame[None]
        return np.transpose(frame, (2, 0, 1))

    def outputs(self, stacked: np.ndarray) -> np.ndarray:
        """The model's output for a stacked observation (channel-first ``uint8``)."""
        batch = np.ascontiguousarray(stacked, dtype=np.uint8)[None]
        (out,) = self._session.run([ONNX_OUTPUTS[self.action_type]], {ONNX_INPUT: batch})
        return out[0]

    def q_values(self, stacked: np.ndarray) -> np.ndarray:
        """Q-values for a stacked observation (discrete policies)."""
        return self.outputs(stacked)

    def _decide(self, stacked: np.ndarray):
        self.last_output = self.outputs(stacked)
        if self.action_type == "discrete":
            return int(np.argmax(self.last_output))
        return float(np.clip(self.last_output[0], -1.0, 1.0))

    def act(self, frame: np.ndarray):
        """Push one preprocessed frame and return the action."""
        frame = self._channel_first(frame)
        if not self._frames:
            for _ in range(self.frame_stack):
                self._frames.append(frame)
        else:
            self._frames.append(frame)
        return self._decide(np.concatenate(list(self._frames), axis=0))

    def act_stacked(self, stacked: np.ndarray):
        """Action for an observation that is already stacked (e.g. from the gym env)."""
        action = self._decide(stacked)
        return np.array([action], dtype=np.float32) if self.action_type == "steering" else action

    def steering(self, action) -> float:
        """The steering value in [-1, 1] (negative = left) of an action."""
        if self.action_type == "discrete":
            return float(self.meta["steering_values"][int(action)])
        return float(np.asarray(action, dtype=np.float64).reshape(-1)[0])

    def wheels(self, action) -> np.ndarray:
        """``[left, right]`` wheel commands in [-1, 1] for an action."""
        return steering_to_wheels(self.steering(action), self.meta["speed"], self.meta["gain"])
