"""Proximal Policy Optimization (PPO) agent with GAE and clipped objective."""

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from networks.models import ActorCritic
from utils.rollout_buffer import RolloutBuffer


class PPOAgent:
    """PPO agent with shared actor-critic network."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Returns (action, log_prob, value)."""
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, value = self.policy(state_t)
            dist = Categorical(probs)
            action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def get_value(self, state: np.ndarray) -> float:
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.policy.get_value(state_t)
        return value.item()

    def update(self) -> Tuple[float, float]:
        """Run PPO update using the collected rollout. Returns (policy_loss, value_loss)."""
        total_ploss = 0.0
        total_vloss = 0.0
        n_batches = 0

        for _ in range(self.n_epochs):
            for states, actions, old_lp, returns, advantages in self.buffer.get_batches(
                self.batch_size, self.device
            ):
                # normalize advantages
                adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                probs, values = self.policy(states)
                values = values.squeeze(-1)
                dist = Categorical(probs)
                new_lp = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values, returns)
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_ploss += policy_loss.item()
                total_vloss += value_loss.item()
                n_batches += 1

        self.buffer.reset()
        denom = max(n_batches, 1)
        return total_ploss / denom, total_vloss / denom

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
