import numpy as np
import pandas as pd
import pytest

from dotenv import load_dotenv

load_dotenv(".env")

from src.util import get_test_config
from src.environments.util import setup_venv_config, setup_venv
from src.environments.env_configs.rewards import *
from src.environments.env_configs.spaces import (
    LinearObservationSpaces,
    LinearObservation,
)
from stable_baselines3 import PPO


import unittest
from unittest.mock import MagicMock
from numpy.testing import assert_allclose


class MockEnv:
    def __init__(self):
        self._get_value = MagicMock()
        self.values = []
        self.norm_inventory = np.array([])
        self.spread = np.array([])
        self.n_envs = 2


class TestPnLReward(unittest.TestCase):
    def test_pnl_reward(self):
        # Create a mock environment
        env = MockEnv()
        env._get_value = MagicMock(
            side_effect=[
                np.array([100]),
                np.array([110]),
                np.array([100]),
                np.array([110]),
            ]
        )  # Mock the _get_value method to return 100 and then 110
        env.norm_inventory = np.array([0.5])

        # Instantiate the PnLReward class with the mock environment
        pnl_reward = PnLReward(env)

        # Test the start_step method
        pnl_reward.start_step()
        self.assertEqual(pnl_reward.value_start, 100)

        # Test the end_step method
        self.assertEqual(
            pnl_reward.end_step(), 10
        )  # Since inventory is not too high, no penalty should be applied

        # Test with high inventory
        env.norm_inventory = np.array([0.9])
        pnl_reward.start_step()
        self.assertEqual(
            pnl_reward.end_step(), -90
        )  # Penalty of 100 should be applied due to high inventory

    def test_pnl_reward_parallel(self):
        env = MockEnv()
        env._get_value = MagicMock(
            side_effect=[
                np.array([100, 100]),
                np.array([110, 110]),
                np.array([100, 100]),
                np.array([110, 110]),
            ]
        )  # Mock the _get_value method to return 100 and then 110
        env.norm_inventory = np.array([0.5, 0.9])

        pnl_reward = PnLReward(env)
        pnl_reward.start_step()
        self.assertEqual(pnl_reward.value_start.tolist(), np.array([100, 100]).tolist())
        self.assertEqual(pnl_reward.end_step().tolist(), [10, -90])


class TestInventoryReward(unittest.TestCase):
    def test_inventory_reward(self):
        env = MockEnv()
        env.norm_inventory = np.array([0.5])

        reward = InventoryReward(env)
        self.assertEqual(reward.end_step(), -0.5)

        env.norm_inventory = np.array([0.9])
        self.assertEqual(reward.end_step(), -0.9)

    def test_inventory_reward_parallel(self):
        env = MockEnv()
        env.norm_inventory = np.array([0.5, 0.9])

        reward = InventoryReward(env)
        self.assertEqual(reward.end_step().tolist(), [-0.5, -0.9])


class TestAssymetricPnLDampening(unittest.TestCase):
    def test_assymetric_pnl_dampening(self):
        env = MockEnv()
        env._get_value = MagicMock(
            side_effect=[
                np.full(3, 100),
                np.full(3, 110),
            ]
        )
        env.norm_inventory = np.array([0.5, 0.7, 0.9])

        reward = AssymetricPnLDampening(env)
        reward.start_step()
        self.assertEqual(reward.value_start.tolist(), [100, 100, 100])

        expected = [8.88888889, 7.44601638, -4.21631001]
        assert_allclose(reward.end_step(), expected)


logger = logging.getLogger(__name__)


class TestRewards(unittest.TestCase):
    def test_multistep_pnl(self):
        env = MockEnv()
        env.values = [100, 110, 120]
        env.norm_inventory = np.array([0.5, 0.9])
        reward = MultistepPnl(env)
        expected = np.array([20, 20]) - np.array([0, 100])
        assert_allclose(reward.end_step(), expected)

    def test_inventory_integral_penalty(self):
        env = MockEnv()
        env.values = [100, 110, 120]
        env.norm_inventory = np.array([0.5, 0.9])
        reward = InventoryIntegralPenalty(env)
        # Add your expected value for this class and assert it similar to the above test.
        # expected = ...
        # assert_allclose(reward.end_step(), expected)

    def test_simple_inventory_pnl_reward(self):
        env = MockEnv()
        env._get_value = MagicMock(
            side_effect=[
                np.array([100, 100]),
                np.array([110, 110]),
            ]
        )
        env.norm_inventory = np.array([0.5, 1])
        reward = SimpleInventoryPnlReward(env)
        reward.start_step()
        assert_allclose(reward.value_start, np.array([100, 100]))
        expected = np.array([2 / 3 * 10, -95])
        assert_allclose(reward.end_step(), expected)

    def test_spread_pnl_reward(self):
        logger.setLevel(logging.DEBUG)
        env = MockEnv()
        env.spread = np.array([10, 20])
        env.norm_inventory = np.array([0.5, 1]).reshape(-1, 1)
        reward = SpreadPnlReward(env)
        expected = np.array([2 / 3 * 10, -90]).reshape(-1, 1)
        assert_allclose(reward.end_step(), expected)


config = get_test_config("test_rewards")


def test_learning_rewards():
    for k, v in reward_dict.items():
        config.env.reward_space = k
        venv = setup_venv_config(config.data, config.env, config.venv)

        assert venv.env.reward_class_type == v
        model = PPO("MlpPolicy", venv, verbose=0)
        model.learn(100)


def test_learning_rewards_parallel():
    for k, v in reward_dict.items():
        config.env_parallel.reward_space = k
        venv = setup_venv_config(config.data, config.env_parallel, config.venv)

        assert venv.env.reward_class_type == v
        model = PPO("MlpPolicy", venv, verbose=0)
        model.learn(100)


def test_setting_reward_params():
    venv = setup_venv_config(config.data, config.env_reward_params, config.venv)
    assert venv.env.reward_params == {"inventory_threshold": 0.5}
    assert venv.env.reward_class.inventory_threshold == 0.5

    clone_env = venv.clone_venv()
    assert clone_env.env.reward_params == {"inventory_threshold": 0.5}
    assert clone_env.env.reward_class.inventory_threshold == 0.5
