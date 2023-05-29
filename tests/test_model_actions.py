import logging
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from src.data_management import get_data_by_dates

from dotenv import load_dotenv

load_dotenv(".env")

from src.util import get_test_config
from src.environments.util import setup_venv_config, setup_venv
from src.environments.env_configs.spaces import (
    LinearObservationSpaces,
    LinearObservation,
)
from stable_baselines3 import PPO
from src.environments.env_configs.rewards import reward_dict
from src.cloning import clone_bc, CloneDuration, load_trained_model
from src.environments.env_configs.policies import ASPolicyVec
from src.model_testing import trained_vs_manual

config = get_test_config("test_model_actions")


def test_model_learn():
    venv = setup_venv_config(config.data, config.env, config.venv)
    model = PPO("MlpPolicy", venv, verbose=0)
    model.learn(total_timesteps=100)


def test_model_all_rewards():
    data = get_data_by_dates(**config.data)
    for k, v in reward_dict.items():
        venv = setup_venv(data, reward_class=v)
        venv.env.reset_metrics_on_reset = False

        try:
            model = PPO("MlpPolicy", venv, verbose=0)
            model.learn(100)
        except Exception as e:
            logging.error(f"Error in {k}: {e}")
            raise e


def test_model_all_rewards_parallel(caplog):
    # caplog.set_level(logging.DEBUG)
    data = get_data_by_dates(**config.data)
    for k, v in reward_dict.items():
        # if k == "inventory_integral_penalty":
        venv = setup_venv(data, reward_class=v, time_envs=3)
        venv.env.reset_metrics_on_reset = False

        try:
            model = PPO("MlpPolicy", venv, verbose=0)
            model.learn(100)
        except Exception as e:
            logging.error(f"Error in {k}: {e}")
            raise e


@pytest.mark.slow
def test_cloning():
    venv = setup_venv_config(config.data, config.env, config.venv)

    model = PPO("MlpPolicy", venv, verbose=0)
    expert_policy = ASPolicyVec(env=venv.env, **config.expert_params)
    action_func = expert_policy.get_action_func()

    obs = venv.reset()
    count = 0
    print(len(venv.env.data))
    # while True:
    #     count += 1
    #     action = action_func(obs)
    #     obs, _, done, _ = venv.step(action)
    #     if done:
    #         break
    print(count)
    # clone_bc(
    #     venv, expert_policy, model, CloneDuration.Short, "test_model", testing=False
    # )
    model = load_trained_model("test_model", venv, False)
    model_metrics, expert_metrics = trained_vs_manual(
        venv,
        model,
    )
    print(model_metrics)
    print(expert_metrics)
