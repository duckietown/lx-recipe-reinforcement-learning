"""Support code for the reinforcement learning LX (provided by the recipe, not edited by learners).

- :mod:`rl_duckiematrix.gridworld`: a tiny Duckietown tile world for the tabular RL notebook.
- :mod:`rl_duckiematrix.env`: the Duckiematrix as a Gymnasium environment, with the wrappers used
  by the deep RL notebooks (:func:`make_env`).
- :mod:`rl_duckiematrix.training`: training loops, evaluation and plotting helpers.
- :mod:`rl_duckiematrix.export`: ONNX export of a trained policy and the :class:`OnnxPolicy` that
  runs it (in the evaluation notebook and on the robot).
- :mod:`rl_duckiematrix.kinematics`: the steering -> wheel commands mapping of a real robot, through
  its kinematics calibration.
"""
