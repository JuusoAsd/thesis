"""
Testing GroupRewardCallback is difficult as it interacts with the repeater class
Ignored for now
"""
import sys, os, logging
import numpy as np
import pandas as pd
import pytest

from dotenv import load_dotenv

load_dotenv(".env")

from src.data_management import get_data_by_dates
from src.util import get_test_config
from src.environments.util import setup_venv_config, setup_venv
from src.environments.env_configs.policies import ASPolicyVec
from src.environments.env_configs.spaces import (
    ActionSpace,
    LinearObservationSpaces,
    LinearObservation,
)
from src.environments.env_configs.callbacks import ExternalMeasureCallback
from src.cloning import load_model_by_config

config = get_test_config("test_env")
test_data = {
    "best_bid": [99, 99, 99],  # best bid price, used to determine market order
    "best_ask": [101, 101, 101],
    "low_price": [50, 50, 50],  # low price, used to determine limit buys
    "high_price": [150, 150, 150],  # high price, used to determine limit sells
}
df = pd.DataFrame(test_data)
df["mid_price"] = (df["best_bid"] + df["best_ask"]) / 2
venv = setup_venv(
    df,
    obs_space=LinearObservation(
        LinearObservationSpaces.OnlyInventorySpace,
    ),
    env_params={"max_ticks": 30},
)
venv.reset()
callback = ExternalMeasureCallback(df.to_numpy(), venv, wait=2, freq=1, patience=1)


class Helper:
    """
    Build a helper class that looks like the model that callback sees but
    returns predetermined values when calling .predict()
    """

    def __init__(self, values) -> None:
        self.values = values
        self.current_step = 0

    def predict(self, obs, deterministic=False):
        self.current_step += 1
        return np.array([self.values[self.current_step - 1]])


def test_get_performance_ok__single():
    metrics = {
        "sharpe": np.array([0.5]),
        "max_inventory": np.array([0.5]),
    }
    performance_metric = callback.get_performance(metrics)
    assert performance_metric == 0.5


def test_get_performance_liquidated_single():
    metrics = {
        "sharpe": np.array([2]),
        "max_inventory": np.array([1]),
    }
    performance_metric = callback.get_performance(metrics)
    assert performance_metric == 0


def test_get_performance_ok_multiple():
    metrics = {
        "sharpe": np.array([2, 0.5]),
        "max_inventory": np.array([0.5, 0.5]),
    }
    performance_metric = callback.get_performance(metrics)
    assert performance_metric == 0.5


def test_get_performance_liquidated_multiple():
    metrics = {
        "sharpe": np.array([2, 0.5]),
        "max_inventory": np.array([1, 0.5]),
    }
    performance_metric = callback.get_performance(metrics)
    assert performance_metric == 0


def test_external_measure_callback(caplog):
    # caplog.set_level(logging.DEBUG)
    venv = setup_venv(
        df,
        obs_space=LinearObservation(
            LinearObservationSpaces.OnlyInventorySpace,
        ),
        env_params={"max_ticks": 30},
    )
    venv.reset()
    callback = ExternalMeasureCallback(df.to_numpy(), venv, wait=2, freq=1, patience=1)
    good_action = np.array([1, 1, -1, 1])
    helper = Helper([good_action, good_action, good_action])
    callback.__setattr__("locals", {"self": helper})

    callback._on_rollout_end()
    assert callback.eval_count == 1
    callback._on_rollout_end()


def test_run_rollout_end():
    config = get_test_config("test_callbacks")
    venv = setup_venv_config(config.data, config.env, config.venv)
    model = load_model_by_config(config, venv)
    data = get_data_by_dates(**config.data)
    callback = ExternalMeasureCallback(
        data.to_numpy(), venv, wait=2, freq=1, patience=1, eval_mode="return/inventory"
    )
    callback.__setattr__("locals", {"self": model})

    callback._on_rollout_end()
    callback._on_rollout_end()

    # trigger first evaluation
    callback._on_rollout_end()
    best = callback.best_reward
    metrics = callback.best_performance_metrics
    assert best == metrics["episode_return"] / (1e-6 + metrics["mean_abs_inv"])


def test_run_rollout_end_parallel():
    config = get_test_config("test_callbacks")
    venv = setup_venv_config(config.data, config.env, config.venv)

    model = load_model_by_config(config, venv)
    data = get_data_by_dates(**config.data)
    callback = ExternalMeasureCallback(
        data.to_numpy(),
        venv,
        wait=2,
        freq=1,
        patience=1,
        eval_mode="min_sharpe",
        **config.parallel_callback
    )
    callback.__setattr__("locals", {"self": model})

    callback._on_rollout_end()
    callback._on_rollout_end()

    # trigger first evaluation
    callback._on_rollout_end()
    best = callback.best_reward
    metrics = callback.best_performance_metrics

    print(best)
    print(metrics)
