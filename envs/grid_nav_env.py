"""2D Grid Navigation Environment for Reinforcement Learning."""

from collections import deque
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


class GridNavEnv(gym.Env):
    """A 2D grid world where a robot navigates from start to goal avoiding obstacles.

    Observation:
        Flattened grid array where 0=free, 1=obstacle, 2=robot, 3=goal.

    Actions:
        0: Up, 1: Down, 2: Left, 3: Right

    Rewards:
        +100  reach the goal
        -10   hit wall or obstacle
        -1    each step
        +1    getting closer to goal (Manhattan distance shaping)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        grid_size: int = 10,
        obstacle_ratio: float = 0.2,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.obstacle_ratio = obstacle_ratio
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(grid_size * grid_size,), dtype=np.float32
        )

        self.grid: Optional[np.ndarray] = None
        self.agent_pos: Optional[Tuple[int, int]] = None
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.steps = 0
        self._prev_dist: Optional[int] = None
        self._fig = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.steps = 0
        self._generate_solvable_map()
        self._prev_dist = self._manhattan()
        return self._get_obs(), {"agent_pos": self.agent_pos, "goal_pos": self.goal_pos}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.steps += 1
        r, c = self.agent_pos
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        nr, nc = r + dr, c + dc

        terminated = False
        truncated = self.steps >= self.max_steps

        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
            reward = -10.0  # hit wall
        elif self.grid[nr, nc] == 1:
            reward = -10.0  # hit obstacle
        else:
            self.agent_pos = (nr, nc)
            if self.agent_pos == self.goal_pos:
                reward = 100.0
                terminated = True
            else:
                curr_dist = self._manhattan()
                # reward shaping: +1 closer, 0 same, -1 farther
                reward = -1.0 + float(self._prev_dist - curr_dist)
                self._prev_dist = curr_dist

        info = {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "success": terminated,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        grid_rgb = self._build_rgb()
        if self.render_mode == "rgb_array":
            return grid_rgb
        if self.render_mode == "human":
            if self._fig is None:
                self._fig, self._ax = plt.subplots(1, 1, figsize=(5, 5))
                plt.ion()
            self._ax.clear()
            self._ax.imshow(grid_rgb)
            self._ax.set_title(f"Step {self.steps}")
            self._ax.axis("off")
            plt.pause(0.05)
        return None

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _generate_solvable_map(self):
        """Generate a random grid that is guaranteed solvable (BFS check)."""
        rng = self.np_random
        while True:
            self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            num_obstacles = int(self.grid_size * self.grid_size * self.obstacle_ratio)
            flat_indices = rng.choice(
                self.grid_size * self.grid_size, num_obstacles, replace=False
            )
            for idx in flat_indices:
                self.grid[idx // self.grid_size, idx % self.grid_size] = 1

            empty = np.argwhere(self.grid == 0)
            if len(empty) < 2:
                continue
            chosen = rng.choice(len(empty), 2, replace=False)
            self.agent_pos = tuple(empty[chosen[0]])
            self.goal_pos = tuple(empty[chosen[1]])

            if self._bfs_reachable():
                break

    def _bfs_reachable(self) -> bool:
        """BFS to verify the goal is reachable from the agent's position."""
        queue = deque([self.agent_pos])
        visited = {self.agent_pos}
        while queue:
            r, c = queue.popleft()
            if (r, c) == self.goal_pos:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.grid_size
                    and 0 <= nc < self.grid_size
                    and self.grid[nr, nc] == 0
                    and (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    # kept for backward compatibility with tests
    is_solvable = _bfs_reachable

    def _manhattan(self) -> int:
        return abs(self.agent_pos[0] - self.goal_pos[0]) + abs(
            self.agent_pos[1] - self.goal_pos[1]
        )

    def _get_obs(self) -> np.ndarray:
        obs = self.grid.copy().astype(np.float32)
        obs[self.agent_pos] = 2.0
        obs[self.goal_pos] = 3.0
        return obs.flatten()

    def _build_rgb(self) -> np.ndarray:
        """Build an RGB image of the current grid state."""
        img = np.full((self.grid_size, self.grid_size, 3), 255, dtype=np.uint8)
        img[self.grid == 1] = [40, 40, 40]       # obstacles – dark grey
        img[self.agent_pos] = [46, 134, 222]      # agent – blue
        img[self.goal_pos] = [39, 174, 96]        # goal – green
        return img
