import pandas as pd
from stable_baselines3 import PPO

from src.training import load_trained_model
from src.environments.mm_env_vec import MMVecEnv
from src.environments.env_configs.policies import ASPolicyVec
from src.environments.env_configs.spaces import ActionSpace, ObservationSpace


base = pd.read_csv("/home/juuso/Documents/gradu/parsed_data/aggregated/base_data.csv")
indicators = pd.read_csv(
    "/home/juuso/Documents/gradu/parsed_data/aggregated/indicator_data.csv"
)
data = pd.merge(base, indicators, on="timestamp", how="left").ffill().dropna()
data["mid_price"] = (data.best_bid + data.best_ask) / 2
column_mapping = {col: n for (n, col) in enumerate(data.columns)}


def run_ml():
    train_timesteps = 10000
    actions = 100_000
    env = MMVecEnv(
        data.to_numpy(),
        n_envs=1,
        params={
            "observation_space": ObservationSpace.OSIObservation,
            "action_space": ActionSpace.NormalizedAction,
        },
        column_mapping=column_mapping,
    )
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=train_timesteps)
    obs = env.reset()
    n = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones or n > actions:
            break

    print(env.get_metrics())


def run_manual():
    actions = 100_000
    env = MMVecEnv(
        data.to_numpy(),
        n_envs=1,
        params={
            "observation_space": ObservationSpace.OSIObservation,
            "action_space": ActionSpace.NormalizedAction,
        },
        column_mapping=column_mapping,
    )
    policy = ASPolicyVec(
        max_order_size=10,
        tick_size=0.0001,
        max_ticks=10,
        price_decimals=4,
        inventory_target=0,
        risk_aversion=0.1,
        order_size=1,
    )
    obs = env.reset()
    for i in range(actions):
        action = policy.get_action(obs)
        obs, rewards, dones, info = env.step(action)

    env.get_metrics()


def run_ml_cloned():
    actions = 100_000
    model = load_trained_model()
    env = MMVecEnv(
        data.to_numpy(),
        n_envs=1,
        params={
            "observation_space": ObservationSpace.OSIObservation,
            "action_space": ActionSpace.NormalizedAction,
        },
        column_mapping=column_mapping,
    )
    obs = env.reset()
    n = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones or n > actions:
            break

    print(env.get_metrics())


if __name__ == "__main__":
    # run_manual()
    # run_ml()
    run_ml_cloned()
