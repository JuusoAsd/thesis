"""
All things related to training the model
"""
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from stable_baselines3 import PPO

from environments.env_configs.spaces import ActionSpace, ObservationSpace
from environments.env_configs.policies import ASPolicyVec
from environments.mm_env_vec import MMVecEnv, SBMMVecEnv

from environments.env_configs.rewards import *

load_dotenv()

base = pd.read_csv(os.getenv("BASE_PATH"))
indicators = pd.read_csv(os.getenv("INDICATOR_PATH"))
data = pd.merge(base, indicators, on="timestamp", how="left").ffill().dropna()
data["mid_price"] = np.round((data.best_bid + data.best_ask) / 2, 5)

column_mapping = {col: n for (n, col) in enumerate(data.columns)}
rng = np.random.default_rng(0)


def run_ml():
    train_timesteps = 100_0
    actions = 100_000
    env = SBMMVecEnv(
        MMVecEnv(
            data.to_numpy(),
            n_envs=5,
            params={
                "observation_space": ObservationSpace.OSIObservation,
                "action_space": ActionSpace.NormalizedAction,
            },
            column_mapping=column_mapping,
            reward_class=InventoryIntegralPenalty,
        )
    )

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=train_timesteps)
    obs = env.reset()
    n = 0
    print("trained")
    env.reset_envs = False
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        n += 1
        if np.all(dones) or n > actions:
            print("done", dones, n)
            break

    print(env.env.get_metrics())


def run_manual():
    # logging.basicConfig(level=logging.DEBUG)
    actions = 100000
    env = MMVecEnv(
        data.to_numpy(),
        n_envs=1,
        params={
            "observation_space": ObservationSpace.OSIObservation,
            "action_space": ActionSpace.NormalizedIntegerAction,
        },
        column_mapping=column_mapping,
        reward_class=InventoryIntegralPenalty,
        action_type="integer",
    )
    policy = ASPolicyVec(
        max_order_size=10,
        tick_size=0.0001,
        max_ticks=10,
        price_decimals=4,
        inventory_target=0,
        risk_aversion=0.1,
        order_size=1,
        action_type="integer",
    )
    obs = env.reset()
    for i in range(actions):
        action = policy.get_integer_action(obs)
        obs, rewards, dones, info = env.step(action)
        if np.all(dones):
            break
    print(env.get_metrics())

    env.save_metrics("manual_policy_integer")


def compare_ml_as():
    actions = 100
    model = load_trained_model()
    env = MMVecEnv(
        data.to_numpy(),
        n_envs=1,
        params={
            "observation_space": ObservationSpace.OSIObservation,
            "action_space": ActionSpace.NormalizedIntegerAction,
        },
        column_mapping=column_mapping,
        reward_class=PnLReward,
        action_type="integer",
    )
    manual = ASPolicyVec(
        max_order_size=10,
        tick_size=0.0001,
        max_ticks=10,
        price_decimals=4,
        inventory_target=0,
        risk_aversion=0.2,
        order_size=1,
        action_type="integer",
    )

    for i in range(5):
        print(env.action_space.sample())

    obs = env.reset()
    n = 0
    while True:
        action_ml, _states = model.predict(obs)
        action_manual = manual.get_integer_action(obs)
        print(f"ml: {np.round(action_ml).tolist()}, manual: {action_manual.tolist()}")
        obs, rewards, dones, info = env.step(action_ml)
        n += 1
        if dones or n > actions:
            break


def run_manual_float():
    # logging.basicConfig(level=logging.DEBUG)
    actions = 100000
    env_cont = MMVecEnv(
        data.to_numpy(),
        n_envs=1,
        params={
            "observation_space": ObservationSpace.OSIObservation,
            "action_space": ActionSpace.NormalizedAction,
        },
        column_mapping=column_mapping,
        reward_class=InventoryIntegralPenalty,
        action_type="continuous",
    )
    policy_cont = ASPolicyVec(
        max_order_size=10,
        tick_size=0.0001,
        max_ticks=10,
        price_decimals=4,
        inventory_target=0,
        risk_aversion=0.1,
        order_size=1,
        action_type="continuous",
    )
    import environments.env_configs.policies as policies

    env_int = MMVecEnv(
        data.to_numpy(),
        n_envs=1,
        params={
            "observation_space": ObservationSpace.OSIObservation,
            "action_space": ActionSpace.NormalizedIntegerAction,
        },
        column_mapping=column_mapping,
        reward_class=InventoryIntegralPenalty,
        action_type="integer",
    )
    policy_int = ASPolicyVec(
        max_order_size=10,
        tick_size=0.0001,
        max_ticks=10,
        price_decimals=4,
        inventory_target=0,
        risk_aversion=0.1,
        order_size=1,
        action_type="integer",
    )

    obs_cont = env_cont.reset()
    obs_int = env_int.reset()

    n = 0
    for i in range(actions):
        action_cont = policy_cont.get_continuous_action(obs_cont)
        obs_cont, _, dones, _ = env_cont.step(action_cont)

        action_int = policy_int.get_integer_action(obs_int)
        obs_int, _, _, _ = env_int.step(action_int)

        n += 1
        if n >= actions or np.all(dones):
            break
    print(env_cont.get_metrics())
    print(env_int.get_metrics())

    env_cont.save_metrics("manual_policy_continuous")
    env_int.save_metrics("manual_policy_integer")


if __name__ == "__main__":
    # clone_manual_policy()
    clone_manual_policy_bc()
    run_ml_cloned()
    compare_ml_as()

    # model = load_trained_model()
    # run_manual()
    # run_ml()
    # run_manual_float()
