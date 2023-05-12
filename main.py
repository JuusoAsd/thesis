import pandas as pd
from stable_baselines3 import PPO
import numpy as np

# from src.training import load_trained_model
# from src.environments.mm_env_vec import MMVecEnv
# from src.environments.env_configs.policies import ASPolicyVec
# from src.environments.env_configs.spaces import ActionSpace, ObservationSpace


# base = pd.read_csv("/home/juuso/Documents/gradu/parsed_data/aggregated/base_data.csv")
# indicators = pd.read_csv(
#     "/home/juuso/Documents/gradu/parsed_data/aggregated/indicator_data.csv"
# )
# data = pd.merge(base, indicators, on="timestamp", how="left").ffill().dropna()
# data["mid_price"] = (data.best_bid + data.best_ask) / 2
# column_mapping = {col: n for (n, col) in enumerate(data.columns)}


# def run_ml():
#     train_timesteps = 10000
#     actions = 100_000
#     env = MMVecEnv(
#         data.to_numpy(),
#         n_envs=1,
#         params={
#             "observation_space": ObservationSpace.OSIObservation,
#             "action_space": ActionSpace.NormalizedAction,
#         },
#         column_mapping=column_mapping,
#     )
#     model = PPO("MlpPolicy", env, verbose=1)
#     model.learn(total_timesteps=train_timesteps)
#     obs = env.reset()
#     n = 0
#     while True:
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = env.step(action)
#         if dones or n > actions:
#             break

#     print(env.get_metrics())


# def run_manual():
#     actions = 100_000
#     env = MMVecEnv(
#         data.to_numpy(),
#         n_envs=1,
#         params={
#             "observation_space": ObservationSpace.OSIObservation,
#             "action_space": ActionSpace.NormalizedAction,
#         },
#         column_mapping=column_mapping,
#     )
#     policy = ASPolicyVec(
#         max_order_size=10,
#         tick_size=0.0001,
#         max_ticks=10,
#         price_decimals=4,
#         inventory_target=0,
#         risk_aversion=0.1,
#         order_size=1,
#     )
#     obs = env.reset()
#     for i in range(actions):
#         action = policy.get_action(obs)
#         obs, rewards, dones, info = env.step(action)

#     env.get_metrics()


# def run_ml_cloned():
#     actions = 100_000
#     model = load_trained_model()
#     env = MMVecEnv(
#         data.to_numpy(),
#         n_envs=1,
#         params={
#             "observation_space": ObservationSpace.OSIObservation,
#             "action_space": ActionSpace.NormalizedAction,
#         },
#         column_mapping=column_mapping,
#     )
#     obs = env.reset()
#     n = 0
#     while True:
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = env.step(action)
#         if dones or n > actions:
#             break

#     print(env.get_metrics())


def curve_func(t, a, k):
    return a * np.exp(-k * t)


def test_fit_curve():
    a = 2
    k = 0.1
    price_levels = np.array(range(1, 1996))
    lambdas = [curve_func(p, a, k) for p in price_levels]

    x = price_levels
    y = np.log(lambdas)
    cov = np.cov(x, y, bias=True)
    var = np.var(x)

    slope = cov[0, 1] / var
    intercept = np.mean(y) - slope * np.mean(x)

    intercept_hat = np.exp(intercept)
    slope_hat = -slope
    print(f"covar: {cov}, var: {var}")
    print(f"slope: {slope}, intercept: {intercept}")
    print(f"intercept_hat: {intercept_hat}, slope_hat: {slope_hat}")
    # print(a1_calc, k_calc)

    #  x = -price_levels
    #     y = np.log(lambdas)
    #     print(np.cov(x, y))
    #     intercept = np.cov(x, y)[0, 1] / np.var(x)
    #     slope = np.mean(y) - intercept * np.mean(x)
    #     slope_exp = np.exp(slope)
    #     print(slope_exp, intercept)


from datetime import datetime, timedelta
from stable_baselines3 import PPO
from model_testing import test_trained_model, test_trained_vs_manual
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
from environments.env_configs.callbacks import ExternalMeasureCallback

if __name__ == "__main__":
    reward = AssymetricPnLDampening
    venv = setup_venv(
        data=get_data_by_dates("2021_12_24"),
        act_space=ActionSpace.NormalizedAction,
        reward_class=reward,
        inv_envs=5,
        time_envs=4,
        env_params={
            "inv_jump": 0.18,
            "data_portion": 0.5,
            "reward_params": {"liquidation_threshold": 0.8},
        },
    )
    model = load_trained_model("clone_bc", venv)
    model.learn(total_timesteps=1_000)
