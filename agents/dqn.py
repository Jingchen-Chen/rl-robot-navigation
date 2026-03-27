"""Deep Q-Network (DQN) agent with experience replay and target network."""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.models import QNetwork
from utils.replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN agent with linear epsilon-greedy decay and soft target updates."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        tau: float = 0.005,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay_steps: int = 10_000,
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.device = device

        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, state_dim, device)

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.steps_done = 0

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

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        current_q = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # soft update target network
        for tp, lp in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)

        return loss.item()

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
