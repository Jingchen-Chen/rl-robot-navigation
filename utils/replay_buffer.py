"""Experience replay buffer for off-policy algorithms (DQN)."""

from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size replay buffer stored as pre-allocated NumPy arrays."""

    def __init__(self, capacity: int, state_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.pos = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.from_numpy(self.states[idx]).to(self.device),
            torch.from_numpy(self.actions[idx]).to(self.device),
            torch.from_numpy(self.rewards[idx]).to(self.device),
            torch.from_numpy(self.next_states[idx]).to(self.device),
            torch.from_numpy(self.dones[idx]).to(self.device),
        )

    def __len__(self) -> int:
        return self.size
