# LX Recipe: Reinforcement Learning

Recipe of the [reinforcement learning LX](https://github.com/duckietown/lx-reinforcement-learning)
(solution: [lx-reinforcement-learning-solution](https://github.com/duckietown/lx-reinforcement-learning-solution)).
Learners train a Duckiebot to follow the lane from camera images with deep Q-learning (and,
optionally, soft actor-critic) in the Duckiematrix (gym mode), export the policy to ONNX and run it on
a real Duckiebot.

## What is in here

| path | purpose |
|---|---|
| `packages/rl_duckiematrix/` | support code the learner does not edit: the gridworld of notebook 01, the Duckiematrix Gymnasium environment and its wrappers (`make_env`), the training loops, ONNX export and the `OnnxPolicy` runner |
| `packages/agent/` | the deployment agent (`python3 -m agent`, started by `launchers/default.sh`): camera -> learner's `preprocess` -> `OnnxPolicy` -> wheel commands at the policy's decision rate (5 Hz), through the Duckietown SDK. On a real robot the steering becomes the simulator's speed and turn rate, then PWM through the robot's kinematics calibration (`rl_duckiematrix/kinematics.py`) |
| `assets/duckiematrix/map/loop/` | the training map: the standard loop with a 160x120 camera (`cameras.yaml`), which makes the simulator step faster than the default 640x480 |
| `Dockerfile` | robot image, `dt-commons` base (no ROS: DTPS only); `packages/` on `PYTHONPATH` |
| `Dockerfile.vscode` | editor image: the robot-side packages plus PyTorch, ONNX and the notebook widgets (`dependencies-vscode.txt`) |
| `settings.yaml` | map and vehicle used by `dts code start_matrix` and `dts code workbench -m` |

The learner's code lives in the LX repository: `packages/solution/` (`q_learning.py`, `preprocessing.py`,
`dqn.py`, `sac.py`) and `assets/models/policy.onnx` (exported by notebook 03 or 05, copied into the
robot image by `dts code build`). The JSON file next to the model says whether it is a Q-network
(`action_type: discrete`) or a steering policy (`steering`); the agent runs both.

## Gym mode

The notebooks need the Duckiematrix in **gym mode** (synchronous stepping): `dts code start_matrix --gym`
(or `dts matrix run --standalone --gym --delta-t 0.1 --no-tutorial --map assets/duckiematrix/map/loop`
from this directory). `start_matrix --gym` takes the time step from `settings.yaml` (`matrix.delta_t`);
it must match `rl_duckiematrix.env.DELTA_T`, which the environment uses to compute speeds.

The engine accepts one client at a time: a notebook must `close()` its environment before
another one connects. `dts code workbench -m` uses the normal real-time mode, as on a real robot.

The Python environment inside the editor finds the engine through `rl_duckiematrix.engine`
(`DUCKIEMATRIX_HOST`, then `127.0.0.1`, `host.docker.internal` and the default gateway): the editor
container is on Docker's bridge network, the engine listens on the laptop.

## Dependencies

`duckietown-messages` 0.1.1 comes from PyPI. `gym-duckiematrix` and `duckietown-sdk` 0.3 are installed
from their `DTSW-7149-Fix-Gym-mode` branches (`dependencies-py3.txt`) until they are released; switch to
the PyPI versions then. PyTorch is only in the editor image (CPU wheels on arm64, CUDA wheels on amd64); the
robot runs the exported model with ONNX Runtime.

## Tuning notes

- Time step: 0.1 s of simulated time per engine step (`--delta-t 0.1`; the engine's default is 0.025),
  action repeat 2, so one decision per 0.2 s (5 Hz). With 0.05 s per decision a DQN learned the state
  value but not the action differences (Q-values within 0.1-0.3 of each other at a scale of 13): each
  action barely moved the robot. The exported policy stores its decision interval and the robot agent
  acts at that rate.
- Speeds are computed from `delta_t`, not from the pose timestamps: in gym mode those are the
  renderer's wall clock, which includes the agent's computation time.
- Reward (per simulator step, `rl_duckiematrix.env.default_reward`): `alignment - 5 * |distance|`
  (alignment = cosine of the heading error), `-10` and termination off the road. It is about 1 per
  step when driving well, so staying on the road is what maximizes the return; there is no speed term
  because the actions fix the speed. `DuckiematrixEnv(reward_fn=...)` swaps it.
- Actions: `make_env(actions="discrete")` (DQN) gives 5 steering values, `actions="steering"` (SAC) one
  value in [-1, 1]; negative = left (the right wheel faster), at command 0.6 (`STEERING_SPEED`,
  `STEERING_GAIN` in `env.py`; the base environment scales commands by 0.4 to PWM), 0.144 m/s. The
  hand-written controller `lane_oracle` (`clip(8 d + 2 angle)` on the simulator's lane state) never
  leaves the road and scores about 0.9 per simulator step (DQN and SAC end up slightly above it);
  driving straight or randomly leaves it within 4-6 s.
- Rates (native macOS renderer, 160x120 camera): ~18 engine steps/s of wall-clock time, i.e. the
  simulation runs ~1.8x faster than real time; ~7 decisions/s during DQN training (gradient steps and
  the random starts included).
- The engine only resets the robot to its spawn pose (half a tile before a curve, in the outer lane,
  where all four curves turn left; the inner lane's curves turn right with a 0.18 m radius instead of
  0.41 m). After each reset `RandomStart` moves the robot to the inner lane on half of the episodes
  (`lane_oracle` drives into a straight tile, then full-left steering until the heading has turned
  180°), lets `lane_oracle` drive a random number of steps (up to 330, about one lap) and then up to
  10 steps of random constant steering; after a time-limit truncation the next episode continues
  where the robot is, so that episodes start all around the loop rather than always before the same
  curve. To be replaced by a reset-to-pose in the gym protocol once the DTSW-7149 PRs are merged.
- `ActionDelay` applies the wheel commands 0 or 1 simulator steps (0 or 0.1 s) late, drawn per
  episode (`make_env(action_delay=...)`), because the real robot's loop (camera, network, decoding)
  is slower than the simulator's, which has no delay.
- Lane geometry: the duckietown-sdk lane calculator placed lanes half a tile off the rendered road and
  ran them the wrong way until the `DTSW-7149-fix-lane-geometry` fix (rewards and the oracle were
  computed against the wrong road). `DuckiematrixEnv` detects an unfixed SDK
  (`MapInterpreter.tile_center` missing) and corrects for it; remove the workaround once the fix is
  released.
- Results with the defaults (laptop, native renderer): DQN (`DQNConfig()`, 15k decisions, 45 min) has
  every episode reach the 500-decision limit from about 6k decisions; SAC (`SACConfig()`, 15k, 50 min)
  from about 3k. Both drive the loop without leaving the road. DQN zig-zags between neighbouring
  actions (steering change 0.55 per decision, Q-value margin between the two best actions ~0.03); SAC
  steers 4.5-5x more smoothly (0.12-0.14; the oracle 0.02). Both stay within about 1 cm of the lane
  centre on average, closer than the oracle (1.8 cm, up to 3 cm outward in every curve: it only reacts,
  the learned policies see the curve coming). Decision hysteresis on the DQN made it worse; continuous
  control is the fix. (These numbers are from training in the outer lane only, without `ActionDelay`.)
- Both lanes and the action delay (the current defaults), SAC (`SACConfig()`, 15k decisions): every
  episode reaches the limit from about 1.8k decisions, as without them. Mean distance to the lane centre
  (3 x 300 decisions per case): outer lane 1.5-1.75 cm, inner lane 2.4-3.2 cm (oracle 1.9 and 2.5 cm).
  The policy trained in the outer lane only stays on the road in the inner lane too, but 3.9-4.7 cm off
  centre there (0.8 cm in the outer lane): the inner lane's right curves trade some outer-lane precision
  for a better worst case. Trained policies drive slightly better with the 0.1 s delay than without it.
- Known SDK issue: `DB21J(...).delta_time` requests `/delta_time/` without its `map/` prefix (404), so
  the time step cannot be read from the engine yet.
