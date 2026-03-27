"""Evaluate a trained DQN or PPO navigation agent."""

import argparse
import json
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from envs.grid_nav_env import GridNavEnv


def run_episode(env: GridNavEnv, agent, algo: str, render: bool = False):
    """Run one episode. Returns (reward, steps, success, frames)."""
    state, _ = env.reset()
    frames = []
    total_reward = 0.0
    steps = 0
    done = truncated = False

    while not (done or truncated):
        if render:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if algo == "dqn":
            action = agent.select_action(state, greedy=True)
        else:
            action, _, _ = agent.select_action(state)

        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    return total_reward, steps, info.get("success", False), frames


def save_gif(frames: list, path: str, fps: int = 10):
    if not frames:
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("off")
    img = ax.imshow(frames[0])
    def update(i):
        img.set_data(frames[i])
        return [img]
    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps)
    anim.save(path, writer="pillow", fps=fps)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained navigation agent")
    parser.add_argument("--algo", type=str, required=True, choices=["dqn", "ppo"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save_trajectories", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    render_mode = "rgb_array" if args.render else None
    env_cfg = {**cfg["environment"], "render_mode": render_mode}
    env = GridNavEnv(**env_cfg)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if args.algo == "dqn":
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device="cpu")
        agent.load(args.model_path)
    else:
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device="cpu")
        agent.load(args.model_path)

    rewards, steps_list, successes = [], [], 0
    trajectories = []
    
    # [新增] 强制使用训练时的 seed 初始化地图
    seed = cfg["training"]["seed"]
    env.reset(seed=seed) 

    for ep in range(args.num_episodes):
        r, s, suc, frames = run_episode(env, agent, args.algo, args.render)
        rewards.append(r)
        steps_list.append(s)
        if suc:
            successes += 1

        if args.render and frames:
            gif_path = f"results/eval_ep{ep}.gif"
            os.makedirs("results", exist_ok=True)
            save_gif(frames, gif_path)

        if args.save_trajectories:
            trajectories.append({"episode": ep, "reward": r, "steps": s, "success": suc})

    print("\n╔══════════════════════════════════╗")
    print("║      Evaluation Results          ║")
    print("╠══════════════════════════════════╣")
    print(f"║  Algorithm:    {args.algo.upper():<17s} ║")
    print(f"║  Episodes:     {args.num_episodes:<17d} ║")
    print(f"║  Mean Reward:  {np.mean(rewards):<17.2f} ║")
    print(f"║  Std Reward:   {np.std(rewards):<17.2f} ║")
    print(f"║  Success Rate: {successes/args.num_episodes*100:<16.1f}% ║")
    print(f"║  Mean Steps:   {np.mean(steps_list):<17.1f} ║")
    print("╚══════════════════════════════════╝")

    if args.save_trajectories:
        os.makedirs("results", exist_ok=True)
        with open("results/trajectories.json", "w") as f:
            json.dump(trajectories, f, indent=2)
        print("Trajectories saved to results/trajectories.json")


if __name__ == "__main__":
    main()
