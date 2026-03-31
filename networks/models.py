"""Neural network architectures for DQN and PPO agents."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def _mlp(sizes, activation=nn.ReLU):
    """Build a simple MLP from a list of layer sizes."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Q-Network for DQN: maps state -> Q-values for each action."""

    def __init__(self, input_size: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )
        # orthogonal init for stable early learning
        gain = nn.init.calculate_gain("relu")
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.constant_(m.bias, 0.0)
        # smaller init on the output layer
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNetwork(nn.Module):
    """Standalone policy network returning a Categorical distribution."""

    def __init__(self, input_size: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> Categorical:
        logits = self.net(x)
        return Categorical(logits=logits)


class ValueNetwork(nn.Module):
    """Standalone value network: state -> scalar V(s)."""

    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
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
        # shared trunk
        self.features = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        # separate heads
        self.actor_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # orthogonal initialisation (standard for PPO)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        logits = self.actor_head(feat)
        value = self.critic_head(feat)
        return logits, value  # return raw logits (not softmax)

    def get_action(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(obs)
        return value


class DuelingConvQNetwork(nn.Module):
    """Dueling CNN Q-Network that processes the grid spatially.

    Uses the same observation split as ConvActorCritic: the first
    grid_size*grid_size*3 elements are reshaped into (3, H, W) for
    Conv2d; remaining elements are extra features (dx, dy, dist).
    Dueling architecture decomposes Q = V(s) + A(s,a) - mean(A).
    """

    def __init__(self, grid_size: int, num_actions: int, extra_features: int = 3):
        super().__init__()
        self.grid_size = grid_size
        self.grid_flat_dim = grid_size * grid_size * 3
        self.extra_features = extra_features

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_out_dim = 64 * grid_size * grid_size

        fused_dim = conv_out_dim + extra_features
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
        )

        # value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

        # orthogonal init
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(m.bias, 0.0)
        # smaller init on output layers
        for stream in [self.value_stream, self.advantage_stream]:
            last = stream[-1]
            nn.init.orthogonal_(last.weight, gain=0.01)
            nn.init.constant_(last.bias, 0.0)

    def _split_obs(self, x: torch.Tensor):
        grid_flat = x[:, : self.grid_flat_dim]
        extra = x[:, self.grid_flat_dim :]
        grid_2d = grid_flat.view(-1, 3, self.grid_size, self.grid_size)
        return grid_2d, extra

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid_2d, extra = self._split_obs(x)
        conv_feat = self.conv(grid_2d)
        fused = self.fuse(torch.cat([conv_feat, extra], dim=1))
        value = self.value_stream(fused)
        advantage = self.advantage_stream(fused)
        # dueling aggregation: Q = V + A - mean(A)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


class ConvActorCritic(nn.Module):
    """CNN-based Actor-Critic for PPO — treats the grid as a 2D image.

    The observation is expected to be a flat vector of shape
    (grid_size * grid_size * 3 + extra_features,). This module reshapes
    the first part into (3, grid_size, grid_size) for Conv2d processing,
    then concatenates the extra features before the heads.
    """

    def __init__(self, grid_size: int, num_actions: int, extra_features: int = 3):
        super().__init__()
        self.grid_size = grid_size
        self.grid_flat_dim = grid_size * grid_size * 3
        self.extra_features = extra_features

        # CNN feature extractor for the grid
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_out_dim = 64 * grid_size * grid_size

        # fuse CNN features with extra features
        fused_dim = conv_out_dim + extra_features
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
        )

        # separate heads
        self.actor_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # orthogonal initialisation
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(m.bias, 0.0)
        # smaller init for output heads
        for head in [self.actor_head, self.critic_head]:
            last = head[-1]
            nn.init.orthogonal_(last.weight, gain=0.01)
            nn.init.constant_(last.bias, 0.0)

    def _split_obs(self, x: torch.Tensor):
        grid_flat = x[:, : self.grid_flat_dim]
        extra = x[:, self.grid_flat_dim :]
        grid_2d = grid_flat.view(-1, 3, self.grid_size, self.grid_size)
        return grid_2d, extra

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_2d, extra = self._split_obs(x)
        conv_feat = self.conv(grid_2d)
        fused = self.fuse(torch.cat([conv_feat, extra], dim=1))
        logits = self.actor_head(fused)
        value = self.critic_head(fused)
        return logits, value

    def get_action(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(obs)
        return value
