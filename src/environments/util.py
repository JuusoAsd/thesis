import logging

from stable_baselines3.common.vec_env import VecNormalize
from environments.mm_env_vec import MMVecEnv, SBMMVecEnv
from environments.env_configs.rewards import *
from environments.env_configs.spaces import (
    ActionSpace,
    LinearObservationSpaces,
    LinearObservation,
)
from ray.tune.search.repeater import Repeater
from ray.tune.search import Searcher

logger = logging.getLogger(__name__)
from typing import Dict, List, Optional
from data_management import get_data_by_dates
from environments.env_configs.rewards import reward_dict


def setup_venv(
    data,
    obs_space=LinearObservation(LinearObservationSpaces.SimpleLinearSpace),
    act_space=ActionSpace.NormalizedAction,
    reward_class=InventoryIntegralPenalty,
    normalize=False,
    time_envs=1,
    inv_envs=1,
    env_params={},
    venv_params={},
):
    column_mapping = {col: n for (n, col) in enumerate(data.columns)}
    env = MMVecEnv(
        data.to_numpy(),
        params={
            "observation_space": obs_space,
            "action_space": act_space,
        },
        column_mapping=column_mapping,
        reward_class=reward_class,
        time_envs=time_envs,
        inv_envs=inv_envs,
        **env_params,
    )
    venv = SBMMVecEnv(env, **venv_params)

    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=100_000)
    return venv


def setup_venv_config(data_config, env_config, venv_config):
    logging.info(f"setting up venv")
    data = get_data_by_dates(**data_config)
    column_mapping = {col: n for (n, col) in enumerate(data.columns)}
    action = ActionSpace[env_config.spaces.action_space]
    if env_config.spaces.observation_space.type == "linear":
        obs = LinearObservation(
            LinearObservationSpaces[env_config.spaces.observation_space.params]
        )
    else:
        raise NotImplementedError

    reward = reward_dict[env_config.reward]

    env = MMVecEnv(
        data=data.to_numpy(),
        column_mapping=column_mapping,
        action_space=action,
        observation_space=obs,
        reward_class=reward,
        **env_config.params,
    )
    venv = SBMMVecEnv(env, **venv_config)
    return venv
