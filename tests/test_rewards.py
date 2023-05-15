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


import unittest
from unittest.mock import MagicMock
from numpy.testing import assert_allclose


class MockEnv:
    def __init__(self):
        self.norm_inventory = None

    def _get_value(self):
        pass  # This method will be mocked later


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
