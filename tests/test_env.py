"""Tests for the GridNavEnv environment."""

import unittest

import gymnasium as gym
import numpy as np

from envs.grid_nav_env import GridNavEnv


class TestGridNavEnv(unittest.TestCase):
    def setUp(self):
        self.env = GridNavEnv(grid_size=10, obstacle_ratio=0.2, max_steps=200)

    def test_env_creation(self):
        self.assertIsInstance(self.env.action_space, gym.spaces.Discrete)
        self.assertEqual(self.env.action_space.n, 4)
        self.assertIsInstance(self.env.observation_space, gym.spaces.Box)
        self.assertEqual(self.env.observation_space.shape, (100,))

    def test_reset_returns_correct_shape(self):
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, (100,))
        self.assertIsInstance(info, dict)
        self.assertIn("agent_pos", info)
        self.assertIn("goal_pos", info)

    def test_step_returns_correct_tuple(self):
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(0)
        self.assertEqual(obs.shape, (100,))
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
        # force agent and goal to be adjacent
        self.env.agent_pos = (0, 0)
        self.env.goal_pos = (0, 1)
        self.env.grid[0, 0] = 0
        self.env.grid[0, 1] = 0

        _, reward, terminated, _, info = self.env.step(3)  # move right
        self.assertTrue(terminated)
        self.assertEqual(reward, 100.0)
        self.assertTrue(info["success"])

    def test_wall_collision_penalty(self):
        self.env.reset()
        self.env.agent_pos = (0, 0)
        _, reward, terminated, _, _ = self.env.step(0)  # move up into wall
        self.assertEqual(reward, -10.0)
        self.assertFalse(terminated)

    def test_observation_contains_agent_and_goal(self):
        obs, _ = self.env.reset()
        self.assertIn(2.0, obs)  # agent marker
        self.assertIn(3.0, obs)  # goal marker

    def test_different_grid_sizes(self):
        for size in [5, 8, 15]:
            env = GridNavEnv(grid_size=size)
            obs, _ = env.reset()
            self.assertEqual(obs.shape, (size * size,))


if __name__ == "__main__":
    unittest.main()
