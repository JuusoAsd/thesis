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

base = pd.read_csv(os.getenv("BASE_PATH"))
indicators = pd.read_csv(os.getenv("INDICATOR_PATH"))
data = pd.merge(base, indicators, on="timestamp", how="left").ffill().dropna()
data["mid_price"] = np.round((data.best_bid + data.best_ask) / 2, 5)

column_mapping = {col: n for (n, col) in enumerate(data.columns)}
rng = np.random.default_rng(0)


def run_continous_manual_policy():
    actions = 100_000
    env = MMVecEnv(
        data.to_numpy(),
        n_envs=1,
        params={
            "observation_space": ObservationSpace.OSIObservation,
            "action_space": ActionSpace.NormalizedIntegerAction,
        },
        column_mapping=column_mapping,
        reward_class=PnLReward,
        action_type="continuous",
    )
    policy = ASPolicyVec(
        max_order_size=10,
        tick_size=0.0001,
        max_ticks=10,
        price_decimals=4,
        inventory_target=0,
        risk_aversion=0.1,
        order_size=1,
        n_env=1,
        action_type="continuous",
    )
    n = 0
    obs = env.reset()
    while True:
        action = policy.get_continuous_action(obs)
        obs, rewards, dones, info = env.step(action)
        # print(action)
        n += 1
        if dones or n > actions:
            break
    env.save_metrics(f"manual_continuous")


if __name__ == "__main__":
    run_continous_manual_policy()
