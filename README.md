# RL Robot Navigation

A reinforcement learning project for training autonomous navigation agents in a 2D grid environment. Implements **DQN** (with Double DQN + Dueling architecture) and **PPO** from scratch using PyTorch.

## Environment

The agent navigates an 8×8 grid with randomly placed obstacles (15% density). The goal is to reach the target while avoiding obstacles and walls within a 200-step budget.

| Property | Value |
|---|---|
| Grid size | 8×8 |
| Obstacle ratio | 15% |
| Max steps/episode | 200 |
| Actions | 4 (up, down, left, right) |

**Reward shaping:** step penalty, collision penalty, and a large positive reward for reaching the goal.

## Algorithms

### DQN
- Double DQN with Dueling Convolutional Q-Network
- Experience replay buffer (100K transitions)
- Soft target network updates (τ = 0.005)
- ε-greedy exploration with linear decay

### PPO
- Clipped surrogate objective
- Rollout buffer (2048 steps)
- Generalized Advantage Estimation (GAE)

## Results

| Algorithm | Best Eval Reward | Final Success Rate | Steps |
|---|---|---|---|
| **DQN** | **96.8** | **100%** | 1,000,000 |
| PPO | 36.0 | 80% | 1,000,000 |

DQN consistently converges to 100% success rate; PPO exhibits higher variance but still learns effective policies.

## Project Structure

```
rl-robot-navigation/
├── agents/
│   ├── dqn.py            # DQN agent (Double DQN + Dueling network)
│   └── ppo.py            # PPO agent
├── envs/
│   └── grid_nav_env.py   # 2D grid navigation environment (Gymnasium)
├── networks/
│   └── models.py         # Q-network and policy network architectures
├── utils/                # Replay buffer, logging helpers
├── configs/
│   └── default.yaml      # Hyperparameter configuration
├── train.py              # Training entry point
├── evaluate.py           # Evaluation & GIF rendering
└── results/              # Saved evaluation GIFs
```

## Setup

```bash
git clone https://github.com/Jingchen-Chen/rl-robot-navigation.git
cd rl-robot-navigation
pip install -r requirements.txt
```

> **Apple Silicon:** Training automatically uses MPS when available. CUDA is used on Linux/Windows with a compatible GPU. Falls back to CPU otherwise.

## Training

```bash
# Train DQN (default)
python train.py --algo dqn --seed 42

# Train PPO
python train.py --algo ppo --seed 42
```

Training logs are written to `runs/` and viewable with TensorBoard:

```bash
tensorboard --logdir runs/
```

Checkpoints are saved to `checkpoints/`.

## Evaluation

```bash
# Evaluate a trained DQN agent (renders GIFs to results/)
python evaluate.py --algo dqn

# Evaluate PPO
python evaluate.py --algo ppo
```

## Configuration

Edit `configs/default.yaml` to adjust hyperparameters:

```yaml
environment:
  grid_size: 8
  obstacle_ratio: 0.15
  max_steps: 200

dqn:
  lr: 1e-4
  gamma: 0.99
  buffer_size: 100000
  batch_size: 64
  ...
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Gymnasium 0.29+
- See `requirements.txt` for full list

## License

MIT
