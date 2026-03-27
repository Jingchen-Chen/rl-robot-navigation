"""Tests for the GridNavEnv environment."""

import unittest

import gymnasium as gym
import numpy as np

from envs.grid_nav_env import GridNavEnv


class TestGridNavEnv(unittest.TestCase):
    def setUp(self):
        self.env = GridNavEnv(grid_size=8, obstacle_ratio=0.15, max_steps=200)

    def test_env_creation(self):
        self.assertIsInstance(self.env.action_space, gym.spaces.Discrete)
        self.assertEqual(self.env.action_space.n, 4)
        self.assertIsInstance(self.env.observation_space, gym.spaces.Box)
        # 3 channels × grid_size² = 3 × 64 = 192
        self.assertEqual(self.env.observation_space.shape, (3 * 8 * 8,))

    def test_reset_returns_correct_shape(self):
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, (3 * 8 * 8,))
        self.assertIsInstance(info, dict)
        self.assertIn("agent_pos", info)
        self.assertIn("goal_pos", info)

    def test_step_returns_correct_tuple(self):
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(0)
        self.assertEqual(obs.shape, (3 * 8 * 8,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_maps_are_solvable(self):
        for _ in range(20):
            self.env.reset()
            self.assertTrue(
                self.env.is_solvable(),
                "Generated map should always be solvable",
            )

    def test_random_episode_terminates(self):
        obs, _ = self.env.reset()
        done = False
        steps = 0
        while not done and steps < 1000:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            steps += 1
        self.assertTrue(done, "Episode should terminate within max_steps or by reaching goal")

    def test_goal_reachable_gives_positive_reward(self):
        """Directly place agent next to goal and step towards it."""
        self.env.reset()
        self.env.agent_pos = (0, 0)
        self.env.goal_pos = (0, 1)
        self.env.grid[0, 0] = 0
        self.env.grid[0, 1] = 0
        self.env._prev_potential = self.env._potential(self.env.agent_pos)

        _, reward, terminated, _, info = self.env.step(3)  # move right
        self.assertTrue(terminated)
        self.assertGreater(reward, 0)
        self.assertTrue(info["success"])

    def test_wall_collision_penalty(self):
        self.env.reset()
        self.env.agent_pos = (0, 0)
        self.env._prev_potential = self.env._potential(self.env.agent_pos)
        _, reward, terminated, _, _ = self.env.step(0)  # move up into wall
        self.assertLess(reward, 0.0)
        self.assertFalse(terminated)

    def test_observation_three_channel(self):
        obs, _ = self.env.reset()
        n = self.env.grid_size
        # agent plane (channel 1) should have exactly one '1'
        agent_plane = obs[n * n: 2 * n * n]
        self.assertAlmostEqual(agent_plane.sum(), 1.0)
        # goal plane (channel 2) should have exactly one '1'
        goal_plane = obs[2 * n * n:]
        self.assertAlmostEqual(goal_plane.sum(), 1.0)

    def test_fixed_map_mode(self):
        env = GridNavEnv(grid_size=8, fixed_map=True)
        obs1, info1 = env.reset()
        obs2, info2 = env.reset()
        # After first reset, subsequent resets keep same goal and start
        self.assertEqual(info1["goal_pos"], info2["goal_pos"])
        self.assertEqual(info1["agent_pos"], info2["agent_pos"])

    def test_different_grid_sizes(self):
        for size in [5, 8, 12]:
            env = GridNavEnv(grid_size=size)
            obs, _ = env.reset()
            self.assertEqual(obs.shape, (3 * size * size,))


if __name__ == "__main__":
    unittest.main()
