import os
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import tempfile
from stable_baselines3 import PPO
import torch as th
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer

from environments.env_configs.spaces import ActionSpace, ObservationSpace
from environments.env_configs.policies import ASPolicyVec
from environments.mm_env_vec import MMVecEnv, SBMMVecEnv
from environments.env_configs.util import ExponentialBetaSchedule, BCEvalCallback

from environments.env_configs.rewards import *
import environments.env_configs.policies as policies

load_dotenv()

print(f"reading data from: {os.getenv('BASE_PATH')}")
base = pd.read_csv(os.getenv("BASE_PATH"))
indicators = pd.read_csv(os.getenv("INDICATOR_PATH"))
data = pd.merge(base, indicators, on="timestamp", how="left").ffill().dropna()
data["mid_price"] = np.round((data.best_bid + data.best_ask) / 2, 5)
print(data)
column_mapping = {col: n for (n, col) in enumerate(data.columns)}

import os
import tempfile
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import torch as th
from stable_baselines3.common.vec_env import VecNormalize

from environments.env_configs.spaces import ActionSpace, ObservationSpace
from environments.env_configs.policies import (
    ASPolicyVec,
    AlwaysSamePolicyVec,
    SimplePolicyVec,
    RoundingPolicyVec,
    NThDigitPolicyVec,
)
from environments.mm_env_vec import MMVecEnv, SBMMVecEnv
from environments.env_configs.util import ExponentialBetaSchedule, BCEvalCallback
from environments.env_configs.rewards import *
import environments.env_configs.policies as policies


def run_manual_policy():
    normalize = False
    obs_space = ObservationSpace.SimpleObservation
    act_space = ActionSpace.NormalizedAction
    expert_policy = ASPolicyVec
    expert_params = {
        "max_order_size": 5,
        "tick_size": 0.0001,
        "max_ticks": 10,
        "price_decimals": 4,
        "inventory_target": 0,
        "risk_aversion": 0.2,
        "order_size": 1,
        "obs_type": ObservationSpace.SimpleObservation,
        "act_type": ActionSpace.NormalizedAction,
    }

    env = MMVecEnv(
        data.to_numpy(),
        n_envs=1,
        params={
            "observation_space": obs_space,
            "action_space": act_space,
        },
        column_mapping=column_mapping,
        reward_class=InventoryIntegralPenalty,  # reward should not matter?
    )

    venv = SBMMVecEnv(env)
    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=100_000)
    expert = expert_policy(env=env, **expert_params)
    action_func = expert.get_action_func()
    obs = venv.reset()

    done = False
    all_val = []
    n_steps = 0
    while not done:
        action = action_func(obs)
        obs, reward, done, info = venv.step(action)
        current_val = obs[0].tolist() + action[0].tolist()
        all_val.append(current_val)
        n_steps += 1
    pd.DataFrame(all_val).to_csv(
        "/Users/juusoahlroos/Documents/own/gradu/src/data_management/manual_acts.csv",
        index=False,
    )
    print(f"n_steps: {n_steps}")


if __name__ == "__main__":
    # run_continous_manual_policy()
    run_manual_policy()
