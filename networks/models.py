"""Neural network architectures for DQN and PPO agents."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class QNetwork(nn.Module):
    """Q-Network for DQN: maps state -> Q-values for each action."""

    def __init__(self, input_size: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNetwork(nn.Module):
    """Standalone policy network (softmax output)."""

    def __init__(self, input_size: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> Categorical:
        logits = self.net(x)
        return Categorical(logits=logits)


class ValueNetwork(nn.Module):
    """Standalone value network: maps state -> scalar V(s)."""

    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """Combined Actor-Critic for PPO with a shared feature extractor."""

    def __init__(self, input_size: int, num_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        action_probs = F.softmax(self.actor_head(feat), dim=-1)
        value = self.critic_head(feat)
        return action_probs, value

    def get_action(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, value = self.forward(obs)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, value = self.forward(obs)
        dist = Categorical(probs)
        return dist.log_prob(actions), dist.entropy(), value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(obs)
        return value
