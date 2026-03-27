# 🤖 RL Robot Navigation

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Deep Reinforcement Learning for 2D Robot Navigation using **DQN** and **PPO**.

A mobile robot learns to navigate from a random start to a goal in a grid world with obstacles.

## Overview

This project implements and compares two Deep RL algorithms for autonomous navigation:

| Feature | Details |
|---------|---------|
| **Environment** | Custom Gymnasium grid world with random obstacles and BFS-verified solvability |
| **DQN** | Experience replay, target network, linear ε-decay, soft updates |
| **PPO** | Shared actor-critic, GAE, clipped surrogate objective, entropy bonus |
| **Logging** | TensorBoard integration for loss, reward, and success rate |
| **Visualization** | Episode rendering (GIF), Q-value heatmaps, training curves |

## Project Structure

```
rl-robot-navigation/
├── configs/
│   └── default.yaml          # Hyperparameter configuration
├── envs/
│   ├── __init__.py
│   └── grid_nav_env.py       # Gymnasium-compatible grid navigation environment
├── networks/
│   ├── __init__.py
│   └── models.py             # QNetwork, ActorCritic, PolicyNetwork, ValueNetwork
├── agents/
│   ├── __init__.py
│   ├── dqn.py                # DQN agent with replay buffer
│   └── ppo.py                # PPO agent with rollout buffer
├── utils/
│   ├── __init__.py
│   ├── replay_buffer.py      # Fixed-size experience replay (NumPy arrays)
│   ├── rollout_buffer.py     # Trajectory buffer with GAE computation
│   └── visualization.py      # Plotting and animation utilities
├── tests/
│   └── test_env.py           # Unit tests for the environment
├── train.py                  # Main training entry point
├── evaluate.py               # Model evaluation and GIF export
├── requirements.txt
├── setup.py
└── LICENSE
```

## Algorithm Details

### Deep Q-Network (DQN)

DQN approximates the optimal action-value function $Q^*(s, a)$ using a neural network.

**Bellman optimality target:**

$$y_i = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

**Loss function (MSE on TD error):**

$$\mathcal{L}(\theta) = \mathbb{E}\left[(y_i - Q(s, a; \theta))^2\right]$$

Key techniques: experience replay buffer, target network (soft update $\tau = 0.01$), linear $\varepsilon$-greedy decay.

### Proximal Policy Optimization (PPO)

PPO optimizes a clipped surrogate objective to ensure stable policy updates.

**Clipped surrogate objective:**

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$.

**Generalized Advantage Estimation (GAE):**

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                     Agent (DQN / PPO)                  │
│  ┌──────────────┐   action   ┌──────────────────────┐  │
│  │ Neural Net   │ ────────►  │ Replay / Rollout     │  │
│  │ (Q / AC)     │ ◄──────── │ Buffer               │  │
│  └──────┬───────┘   batch   └──────────────────────┘  │
│         │ state                                        │
└─────────┼──────────────────────────────────────────────┘
          │ action ▼          ▲ (state, reward, done)
┌─────────┴──────────────────────────────────────────────┐
│                   GridNavEnv                           │
│  ┌─────┬─────┬─────┬─────┐                            │
│  │  S  │     │  #  │     │   S = Start (agent)        │
│  ├─────┼─────┼─────┼─────┤   G = Goal                 │
│  │     │  #  │     │     │   # = Obstacle              │
│  ├─────┼─────┼─────┼─────┤                            │
│  │  #  │     │     │  #  │   Actions: ↑ ↓ ← →         │
│  ├─────┼─────┼─────┼─────┤   Obs: 3-channel flat vec  │
│  │     │     │  #  │  G  │   Rewards: +100, -5, -1    │
│  └─────┴─────┴─────┴─────┘                            │
└────────────────────────────────────────────────────────┘
```

## Installation

```bash
git clone https://github.com/Jingchen-Chen/rl-robot-navigation.git
cd rl-robot-navigation
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Train DQN agent (600 episodes)
python train.py --algo dqn

# Train PPO agent (300k timesteps)
python train.py --algo ppo

# Custom settings
python train.py --algo dqn --seed 123 --episodes 2000 --device cuda
```

### Evaluation

```bash
# Evaluate trained DQN model
python evaluate.py --algo dqn --model_path checkpoints/best_dqn.pth --render

# Evaluate PPO with trajectory saving
python evaluate.py --algo ppo --model_path checkpoints/best_ppo.pth --save_trajectories
```

### TensorBoard

```bash
tensorboard --logdir runs/
```

### Tests

```bash
python -m pytest tests/ -v
```

## Environment Details

| Property | Value |
|----------|-------|
| Grid size | 8×8 (configurable) |
| Obstacle ratio | 15% (configurable) |
| Observation | 3-channel flattened vector (192-dim): obstacle / agent / goal planes |
| Actions | Discrete(4): Up, Down, Left, Right |
| Reward: reach goal | +100 |
| Reward: hit wall/obstacle | −5 + distance shaping |
| Reward: each step | −1 + distance shaping |
| Max steps | 200 |
| Map generation | Random with BFS solvability guarantee |

## Configuration

All hyperparameters are in [`configs/default.yaml`](configs/default.yaml). Key settings:

| Parameter | DQN | PPO |
|-----------|-----|-----|
| Learning rate | 5e-4 | 3e-4 |
| Discount (γ) | 0.99 | 0.99 |
| Batch size | 64 | 128 |
| Buffer size | 50,000 | 1,024 (rollout) |
| ε-decay steps | 30,000 | — |
| Clip ε | — | 0.2 |
| GAE λ | — | 0.95 |
| Entropy coef | — | 0.02 |
| Update epochs | — | 8 |

## Results

Evaluated on 20 held-out episodes after training (seed 42, 8×8 grid, 15% obstacles).

| Algorithm | Training Budget | Mean Reward | Std Reward | Success Rate | Mean Steps |
|-----------|----------------|-------------|------------|--------------|------------|
| **DQN** | 600 episodes | **90.79** | 0.00 | **100%** | **11.0** |
| **PPO** | 300k timesteps | 50.94 | 35.06 | 100% | 26.6 |

**Key observations:**

- **DQN** converges faster and more stably on this environment, reaching 100% success by episode ~50 and consistently finding near-optimal paths (avg 11 steps vs. BFS-optimal ~9–12).
- **PPO** also achieves 100% success but with higher variance and longer paths, reflecting the on-policy sample efficiency gap on a relatively small discrete action space.
- PPO training exhibits periodic reward drops (catastrophic forgetting pattern), while DQN's off-policy replay buffer provides more stable gradient updates.

> Reproduce results: `python train.py --algo dqn --seed 42` then `python evaluate.py --algo dqn --model_path checkpoints/best_dqn.pth`

## License

[MIT](LICENSE) © 2026 Jingchen Chen

## Acknowledgments

- [Gymnasium](https://gymnasium.farama.org/) — RL environment toolkit
- [PyTorch](https://pytorch.org/) — deep learning framework