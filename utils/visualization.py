"""Visualization utilities for training analysis and episode rendering."""

import os
from typing import Any, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_curves(rewards: list, save_path: Optional[str] = None):
    """Plot episode reward curve with a rolling average."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.3, label="Episode Reward")
    if len(rewards) >= 20:
        window = min(50, len(rewards) // 4)
        avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), avg, label=f"Rolling Avg ({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def render_episode(
    env: Any,
    agent: Any,
    algo: str = "dqn",
    save_path: Optional[str] = None,
    fps: int = 10,
):
    """Render a single episode as a matplotlib animation, optionally saved as GIF."""
    state, _ = env.reset()
    frames = []
    done = truncated = False

    while not (done or truncated):
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if algo == "dqn":
            action = agent.select_action(state, greedy=True)
        else:
            action, _, _ = agent.select_action(state)
        state, _, done, truncated, _ = env.step(action)

    if not frames:
        print("No frames captured.")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    img = ax.imshow(frames[0])

    def update(i):
        img.set_data(frames[i])
        return [img]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps)
    if save_path:
        anim.save(save_path, writer="pillow", fps=fps)
        print(f"Saved animation → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_q_values(agent: Any, env: Any, save_path: Optional[str] = None):
    """Visualize DQN Q-values as heatmaps over the grid."""
    gs = env.grid_size
    action_names = ["Up", "Down", "Left", "Right"]
    q_map = np.zeros((gs, gs, 4))

    for r in range(gs):
        for c in range(gs):
            obs = env.grid.copy().astype(np.float32)
            obs[r, c] = 2.0
            obs[env.goal_pos] = 3.0
            obs_flat = obs.flatten()
            state_t = torch.from_numpy(obs_flat).float().unsqueeze(0)
            with torch.no_grad():
                q_vals = agent.q_net(state_t).cpu().numpy()[0]
            q_map[r, c] = q_vals

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, ax in enumerate(axes):
        im = ax.imshow(q_map[:, :, i], cmap="viridis")
        ax.set_title(action_names[i])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Q-Value Heatmaps", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def compare_algorithms(
    dqn_rewards: list,
    ppo_rewards: list,
    save_path: Optional[str] = None,
):
    """Plot DQN vs PPO training curves side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, name in [(ax1, dqn_rewards, "DQN"), (ax2, ppo_rewards, "PPO")]:
        ax.plot(data, alpha=0.3, label="Raw")
        if len(data) >= 20:
            w = min(50, len(data) // 4)
            avg = np.convolve(data, np.ones(w) / w, mode="valid")
            ax.plot(range(w - 1, len(data)), avg, label=f"Avg ({w})")
        ax.set_title(name)
        ax.set_xlabel("Episode / Update")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Algorithm Comparison: DQN vs PPO", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
