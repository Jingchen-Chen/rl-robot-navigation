"""Deep Q-Network (DQN) agent with Double DQN, Dueling architecture, and experience replay."""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.models import QNetwork, DuelingConvQNetwork
from utils.replay_buffer import ReplayBuffer


class DQNAgent:
    """Double DQN agent with linear epsilon-greedy decay and periodic hard target updates."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        tau: float = 0.005,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay_steps: int = 10_000,
        double_dqn: bool = True,
        updates_per_step: int = 1,
        target_update_interval: int = 0,
        device: str = "cpu",
        grid_size: int = 0,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.double_dqn = double_dqn
        self.updates_per_step = updates_per_step
        self.target_update_interval = target_update_interval
        self.device = device

        if grid_size > 0:
            extra_features = state_dim - grid_size * grid_size * 3
            self.q_net = DuelingConvQNetwork(grid_size, action_dim, extra_features).to(device)
            self.target_net = DuelingConvQNetwork(grid_size, action_dim, extra_features).to(device)
        else:
            self.q_net = QNetwork(state_dim, action_dim).to(device)
            self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, state_dim, device)

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.steps_done = 0
        self._update_count = 0

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.steps_done / self.eps_decay_steps)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.q_net(state_t)
        return q_vals.argmax(dim=1).item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.memory.add(state, action, reward, next_state, done)
        self.steps_done += 1

    def update(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        total_loss = 0.0
        for _ in range(self.updates_per_step):
            states, actions, rewards, next_states, dones = self.memory.sample(
                self.batch_size
            )

            current_q = self.q_net(states).gather(1, actions)

            with torch.no_grad():
                if self.double_dqn:
                    best_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                    max_next_q = self.target_net(next_states).gather(1, best_actions)
                else:
                    max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
                target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

            loss = nn.SmoothL1Loss()(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
            self.optimizer.step()
            total_loss += loss.item()
            self._update_count += 1

        # target network sync: hard copy if interval set, otherwise soft update
        if self.target_update_interval > 0:
            if self._update_count % self.target_update_interval == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            for tp, lp in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)

        return total_loss / self.updates_per_step

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done = ckpt.get("steps_done", 0)

    def set_epsilon(self, eps: float):
        """Manually override epsilon (clamp to [eps_end, 1.0])."""
        eps = float(np.clip(eps, self.eps_end, 1.0))
        if eps <= self.eps_end:
            self.steps_done = self.eps_decay_steps
        else:
            frac = (eps - self.eps_start) / (self.eps_end - self.eps_start)
            self.steps_done = int(frac * self.eps_decay_steps)
