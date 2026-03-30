"""Training script for DQN and PPO robot navigation agents."""

import argparse
import os
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from envs.grid_nav_env import GridNavEnv


def get_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def evaluate(env: GridNavEnv, agent, algo: str, num_episodes: int = 10) -> tuple:
    """Run evaluation episodes. Returns (mean_reward, success_rate)."""
    rewards, successes = [], 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = truncated = False
        while not (done or truncated):
            if algo == "dqn":
                action = agent.select_action(state, greedy=True)
            else:
                action, _, _ = agent.select_action(state)
            state, reward, done, truncated, info = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
        if info.get("success", False):
            successes += 1
    return float(np.mean(rewards)), successes / num_episodes


# ──────────────────────────────────────────────────────────────────
# DQN Training
# ──────────────────────────────────────────────────────────────────

def train_dqn(env: GridNavEnv, cfg: dict, device: str, writer: SummaryWriter):
    dqn_cfg = cfg["dqn"]
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    warmup = dqn_cfg.get("warmup_steps", 2000)

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=dqn_cfg["lr"],
        gamma=dqn_cfg["gamma"],
        buffer_size=int(dqn_cfg["buffer_size"]),
        batch_size=dqn_cfg["batch_size"],
        tau=dqn_cfg["tau"],
        eps_start=dqn_cfg["eps_start"],
        eps_end=dqn_cfg["eps_end"],
        eps_decay_steps=dqn_cfg["eps_decay_steps"],
        device=device,
    )

    best_reward = -float("inf")
    total_steps = 0
    recent_rewards = []

    seed = cfg["training"]["seed"]
    for ep in range(1, dqn_cfg["total_episodes"] + 1):
        # 只有在第一个 episode 时传入 seed 以初始化固定的地图
        if ep == 1:
            state, _ = env.reset(seed=seed)
        else:
            state, _ = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        ep_losses = []
        done = truncated = False

        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            ep_steps += 1
            total_steps += 1

            if total_steps > warmup:
                loss = agent.update()
                if loss is not None:
                    ep_losses.append(loss)

        recent_rewards.append(ep_reward)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)
        avg_reward = np.mean(recent_rewards)

        writer.add_scalar("Reward/Episode", ep_reward, ep)
        writer.add_scalar("Reward/Rolling50", avg_reward, ep)
        writer.add_scalar("Steps/Episode", ep_steps, ep)
        writer.add_scalar("Epsilon", agent.epsilon, ep)
        if ep_losses:
            writer.add_scalar("Loss/DQN", np.mean(ep_losses), ep)

        if ep % 10 == 0:
            avg_loss = f"{np.mean(ep_losses):.4f}" if ep_losses else "warmup"
            print(
                f"[DQN] Ep {ep:>4d}/{dqn_cfg['total_episodes']}  "
                f"R={ep_reward:>7.1f}  Avg50={avg_reward:>7.1f}  "
                f"Steps={ep_steps:>4d}  eps={agent.epsilon:.3f}  loss={avg_loss}"
            )

        if ep % dqn_cfg["eval_interval"] == 0:
            mean_r, suc = evaluate(env, agent, "dqn")
            print(f"       ➜ Eval  mean_R={mean_r:.1f}  success={suc*100:.0f}%")
            writer.add_scalar("Reward/Eval", mean_r, ep)
            writer.add_scalar("Success/Eval", suc, ep)
            if mean_r > best_reward:
                best_reward = mean_r
                agent.save(os.path.join(cfg["training"]["save_dir"], "best_dqn.pth"))

    agent.save(os.path.join(cfg["training"]["save_dir"], "final_dqn.pth"))
    print(f"\n[DQN] Training complete. Best eval reward: {best_reward:.1f}")


# ──────────────────────────────────────────────────────────────────
# PPO Training
# ──────────────────────────────────────────────────────────────────

def train_ppo(env: GridNavEnv, cfg: dict, device: str, writer: SummaryWriter):
    ppo_cfg = cfg["ppo"]
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    grid_size = cfg["environment"]["grid_size"]
    initial_lr = ppo_cfg["lr"]
    initial_ent = ppo_cfg["entropy_coef"]
    ent_end = ppo_cfg.get("entropy_coef_end", 0.005)

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=initial_lr,
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_epsilon=ppo_cfg["clip_epsilon"],
        entropy_coef=initial_ent,
        value_coef=ppo_cfg["value_coef"],
        n_epochs=ppo_cfg["n_epochs"],
        batch_size=ppo_cfg["batch_size"],
        device=device,
        grid_size=grid_size,
    )

    best_reward = -float("inf")
    total_steps = 0
    total_timesteps = ppo_cfg["total_timesteps"]
    rollout_steps = ppo_cfg["rollout_steps"]
    recent_rewards: list = []

    state, _ = env.reset(seed=cfg["training"]["seed"])
    ep_reward = 0.0

    while total_steps < total_timesteps:
        agent.buffer.reset()

        # linear LR and entropy annealing
        frac = 1.0 - total_steps / total_timesteps
        cur_lr = initial_lr * frac
        for pg in agent.optimizer.param_groups:
            pg["lr"] = cur_lr
        agent.entropy_coef = ent_end + (initial_ent - ent_end) * frac

        for _ in range(rollout_steps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.buffer.add(state, action, log_prob, reward, value, done or truncated)
            state = next_state
            ep_reward += reward
            total_steps += 1

            if done or truncated:
                recent_rewards.append(ep_reward)
                if len(recent_rewards) > 20:
                    recent_rewards.pop(0)
                ep_reward = 0.0
                state, _ = env.reset()

        last_value = agent.get_value(state)
        agent.buffer.compute_returns_and_advantages(
            last_value, ppo_cfg["gamma"], ppo_cfg["gae_lambda"]
        )
        p_loss, v_loss = agent.update()

        writer.add_scalar("Loss/Policy", p_loss, total_steps)
        writer.add_scalar("Loss/Value", v_loss, total_steps)
        writer.add_scalar("LR", cur_lr, total_steps)
        if recent_rewards:
            avg_r = np.mean(recent_rewards)
            writer.add_scalar("Reward/Rolling20", avg_r, total_steps)

        pct = total_steps / total_timesteps * 100
        avg_str = f"{np.mean(recent_rewards):.1f}" if recent_rewards else "n/a"
        print(
            f"[PPO] {total_steps:>7d}/{total_timesteps} ({pct:4.1f}%)  "
            f"AvgR={avg_str:>7s}  p_loss={p_loss:.4f}  v_loss={v_loss:.4f}"
        )

        if total_steps % ppo_cfg["eval_interval"] < rollout_steps:
            mean_r, suc = evaluate(env, agent, "ppo")
            print(f"       ➜ Eval  mean_R={mean_r:.1f}  success={suc*100:.0f}%")
            writer.add_scalar("Reward/Eval", mean_r, total_steps)
            writer.add_scalar("Success/Eval", suc, total_steps)
            if mean_r > best_reward:
                best_reward = mean_r
                agent.save(os.path.join(cfg["training"]["save_dir"], "best_ppo.pth"))

    agent.save(os.path.join(cfg["training"]["save_dir"], "final_ppo.pth"))
    print(f"\n[PPO] Training complete. Best eval reward: {best_reward:.1f}")


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train RL navigation agent")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--episodes", type=int, default=None, help="DQN episodes override")
    parser.add_argument("--timesteps", type=int, default=None, help="PPO timesteps override")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.episodes is not None:
        cfg["dqn"]["total_episodes"] = args.episodes
    if args.timesteps is not None:
        cfg["ppo"]["total_timesteps"] = args.timesteps

    seed = cfg["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device(args.device)
    print(f"Device: {device}  |  Algo: {args.algo}  |  Seed: {seed}")

    os.makedirs(cfg["training"]["log_dir"], exist_ok=True)
    os.makedirs(cfg["training"]["save_dir"], exist_ok=True)

    env = GridNavEnv(**cfg["environment"])
    run_name = f"{args.algo}_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join(cfg["training"]["log_dir"], run_name))

    if args.algo == "dqn":
        train_dqn(env, cfg, device, writer)
    else:
        train_ppo(env, cfg, device, writer)

    writer.close()
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
