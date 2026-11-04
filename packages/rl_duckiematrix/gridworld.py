"""A tiny Duckietown tile world for the tabular RL notebook (no simulator needed).

The robot moves on a grid of tiles. Road tiles are drivable; grass is not (driving onto it ends
the episode with a penalty). One tile holds a lost duckie: reaching it ends the episode with a
reward. The dynamics are optionally stochastic: with probability ``slip`` the robot moves in a
random direction instead of the chosen one (wheel slip on the mat).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# tile types
GRASS, ROAD, START, DUCKIE = 0, 1, 2, 3
_TILE_CHARS = {".": GRASS, "#": ROAD, "S": START, "D": DUCKIE}

DEFAULT_MAP = [
    "S####",
    "....#",
    ".###D",
    ".#...",
    ".####",
]

ACTIONS = {0: "up", 1: "right", 2: "down", 3: "left"}
_MOVES = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
_ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


class DuckieGridWorld(gym.Env):
    """Grid world with a Duckietown look. States are integers ``row * width + col``.

    Args:
        layout: rows of ``.`` (grass), ``#`` (road), ``S`` (start) and ``D`` (duckie).
        slip: probability that a move goes in a uniformly random direction instead.
        step_reward: reward of every move (negative: be quick).
        grass_reward: reward for driving onto the grass (ends the episode).
        duckie_reward: reward for reaching the duckie (ends the episode).

    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(  # noqa: PLR0913
        self,
        layout: list[str] | None = None,
        *,
        slip: float = 0.0,
        step_reward: float = -1.0,
        grass_reward: float = -10.0,
        duckie_reward: float = 20.0,
    ) -> None:
        layout = layout or DEFAULT_MAP
        self.tiles = np.array([[_TILE_CHARS[c] for c in row] for row in layout], dtype=np.int64)
        self.height, self.width = self.tiles.shape
        self.slip = slip
        self.step_reward = step_reward
        self.grass_reward = grass_reward
        self.duckie_reward = duckie_reward
        self.start = tuple(int(v) for v in np.argwhere(self.tiles == START)[0])
        self.observation_space = spaces.Discrete(self.height * self.width)
        self.action_space = spaces.Discrete(4)
        self._pos = self.start

    @property
    def n_states(self) -> int:
        return self.height * self.width

    @property
    def n_actions(self) -> int:
        return 4

    def state_of(self, row: int, col: int) -> int:
        return row * self.width + col

    def position_of(self, state: int) -> tuple[int, int]:
        return divmod(int(state), self.width)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        self._pos = self.start
        return self.state_of(*self._pos), {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        action = int(action)
        if self.slip > 0 and self.np_random.random() < self.slip:
            action = int(self.np_random.integers(0, 4))
        dr, dc = _MOVES[action]
        row = min(max(self._pos[0] + dr, 0), self.height - 1)
        col = min(max(self._pos[1] + dc, 0), self.width - 1)
        self._pos = (row, col)
        tile = self.tiles[row, col]
        if tile == GRASS:
            return self.state_of(row, col), self.grass_reward, True, False, {"event": "grass"}
        if tile == DUCKIE:
            return self.state_of(row, col), self.duckie_reward, True, False, {"event": "duckie"}
        return self.state_of(row, col), self.step_reward, False, False, {}

    # ------------------------------------------------------------------ rendering (matplotlib)

    def _tile_colors(self) -> np.ndarray:
        colors = {
            GRASS: (0.36, 0.63, 0.29),
            ROAD: (0.20, 0.20, 0.22),
            START: (0.20, 0.20, 0.22),
            DUCKIE: (0.20, 0.20, 0.22),
        }
        image = np.zeros((self.height, self.width, 3))
        for t, c in colors.items():
            image[self.tiles == t] = c
        return image

    def draw(self, ax=None, *, values: np.ndarray | None = None, policy: np.ndarray | None = None, title: str | None = None):
        """Draw the map; optionally overlay state values (numbers) and a greedy policy (arrows)."""
        import matplotlib.pyplot as plt  # noqa: PLC0415

        if ax is None:
            _, ax = plt.subplots(figsize=(1.2 * self.width, 1.2 * self.height))
        ax.imshow(self._tile_colors(), interpolation="nearest")
        for r in range(self.height):
            for c in range(self.width):
                t = self.tiles[r, c]
                if t == DUCKIE:
                    # a duckie: yellow body, orange beak
                    ax.plot(c, r - 0.05, "o", color="#ffd400", markersize=22, markeredgecolor="k")
                    ax.plot(c + 0.18, r - 0.02, "o", color="#ff8c00", markersize=7)
                    ax.text(c, r + 0.38, "duckie", ha="center", va="center", fontsize=8, color="w")
                elif t == START:
                    ax.text(c, r - 0.3, "start", ha="center", va="center", fontsize=8, color="w")
                if t == GRASS:
                    continue
                s = self.state_of(r, c)
                if values is not None and t != DUCKIE:
                    ax.text(c, r + 0.33, f"{values[s]:.1f}", ha="center", va="center", fontsize=8, color="w")
                if policy is not None and t != DUCKIE:
                    ax.text(c, r, _ARROWS[int(policy[s])], ha="center", va="center", fontsize=16, color="y")
        if self._pos != self.start or values is None:
            # the Duckiebot: a blue square
            ax.plot(self._pos[1], self._pos[0], "s", color="#1f77b4", markersize=12, markeredgecolor="w")
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title)
        return ax

    def render(self):
        """Return an RGB image of the map (``rgb_array`` render mode)."""
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots(figsize=(1.2 * self.width, 1.2 * self.height))
        self.draw(ax)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return image
