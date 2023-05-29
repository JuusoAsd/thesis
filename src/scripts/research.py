"""
File that is used to run experiments to understand model behavior
"""
import time
import os


from dotenv import load_dotenv

from stable_baselines3 import PPO


from src.environments.env_configs.policies import ASPolicyVec
from src.environments.env_configs.rewards import *
from src.environments.util import setup_venv_config
from src.model_testing import trained_vs_manual
from src.data_management import get_data_by_dates
from src.util import get_config, create_config_hash
from src.cloning import CloneDuration, clone_bc, load_trained_model
import os
import logging
from itertools import product
import logging
import pandas as pd
import numpy as np
import gym
from src.environments.env_configs.rewards import BaseRewardClass
import src.environments.env_configs.policies as policies
from src.environments.env_configs.spaces import (
    ActionSpace,
    ObservationSpace,
    LinearObservation,
    LinearObservationSpaces,
)
from stable_baselines3.common.vec_env import VecEnv
from src.environments.env_configs.policies import ASPolicyVec

load_dotenv()

config = get_config("decision_grid", subdirectory=["research_configs"])


def create_model_decision_grid():
    """
    Save a grid of observations and respective actions to understand model behavior
    """

    venv = setup_venv_config(config.data, config.decision_grid.env, config.venv)

    if not isinstance(venv.env.obs_space, LinearObservation):
        raise ValueError(
            f"Must use LinearObservation space for decision grid, currently {venv.obs_space}"
        )

    # because using linear observations, we know that everything is normalized from x to y
    obs_grid = venv.env.obs_space.get_grid_space(
        config.decision_grid.distinct_count_per_variable,
        constant_values=config.decision_grid.constant_values,
    )
    print(f"OBSERVATIONS")
    print(obs_grid)
    # get a model
    model = load_trained_model(
        config.decision_grid.model_name,
        venv,
        normalize=False,
        model_kwargs=config.decision_grid.model_kwargs,
    )
    print("ACTIONS")
    act, _ = model.predict(obs_grid, deterministic=True)
    print(act)