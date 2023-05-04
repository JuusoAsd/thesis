import os
import gym
import numpy as np
from functools import partial
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air.config import RunConfig
from ray.tune.search.repeater import Repeater, TRIAL_INDEX
from ray.tune.search.hyperopt import HyperOptSearch

from datetime import datetime, timedelta
from stable_baselines3 import PPO
from testing import test_trained_model, test_trained_vs_manual
from environments.util import setup_venv
from data_management import get_data_by_dates
from cloning import load_trained_model, save_trained_model
from environments.env_configs.spaces import ActionSpace
from environments.env_configs.rewards import (
    PnLReward,
    AssymetricPnLDampening,
    InventoryIntegralPenalty,
    MultistepPnl,
    InventoryReward,
    SimpleInventoryPnlReward,
    SpreadPnlReward,
)
from environments.env_configs.callbacks import (
    ExternalMeasureCallback,
    GroupRewardCallback,
)
from ray.tune.search.bayesopt import BayesOptSearch
from util import set_seeds, get_config
import random

from ray.tune.schedulers import AsyncHyperBandScheduler
import time


def clone_objective(config):
    print("Running objective function")
    print(config)
    time.sleep(20)
    return {"res": random.randint(0, 100)}


def main_func():
    search_algo = HyperOptSearch(
        mode="max",
    )
    search_space = {
        "x": tune.choice([0, 1, 2, 3, 4]),
        "y": tune.choice([0, 1, 2, 3, 4]),
        "z": 42,
        "w": {"a": 1, "b": 2},
    }
    # repeater = Repeater(search_algo, set_index=True, repeat=5)

    tuner = tune.Tuner(
        trainable=clone_objective,
        param_space=search_space,
        run_config=RunConfig(
            name="test_trial",
            local_dir=os.getenv("COMMON_PATH"),
        ),
        tune_config=tune.TuneConfig(num_samples=20),
    )
    tuner.fit()


if __name__ == "__main__":
    main_func()
