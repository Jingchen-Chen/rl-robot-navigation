"""Rollout buffer for on-policy algorithms (PPO)."""

from typing import Generator, Tuple

import numpy as np
import torch


class RolloutBuffer:
    """Stores trajectories for PPO and computes GAE advantages."""

    def __init__(self):
        self.states: list = []
        self.actions: list = []
        self.log_probs: list = []
        self.rewards: list = []
        self.values: list = []
        self.dones: list = []

        self.returns: np.ndarray = np.array([])
        self.advantages: np.ndarray = np.array([])

    def add(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95
    ):
        """Compute GAE advantages and discounted returns."""
        n = len(self.rewards)
        self.returns = np.zeros(n, dtype=np.float32)
        self.advantages = np.zeros(n, dtype=np.float32)

        gae = 0.0
        next_value = last_value
        for t in reversed(range(n)):
            mask = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]
            next_value = self.values[t]

    def get_batches(
        self, batch_size: int, device: str = "cpu"
    ) -> Generator[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        None,
        None,
    ]:
        """Yield shuffled mini-batches as torch tensors."""
        n = len(self.states)
        indices = np.random.permutation(n)

        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int64)
        old_log_probs = np.array(self.log_probs, dtype=np.float32)

        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield (
                torch.from_numpy(states[idx]).to(device),
                torch.from_numpy(actions[idx]).to(device),
                torch.from_numpy(old_log_probs[idx]).to(device),
                torch.from_numpy(self.returns[idx]).to(device),
                torch.from_numpy(self.advantages[idx]).to(device),
            )

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.returns = np.array([])
        self.advantages = np.array([])

    def __len__(self) -> int:
        return len(self.states)
