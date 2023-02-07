import logging
from gym.wrappers.normalize import NormalizeObservation

from src.environments.mm_env import MMEnv
from src.environments.policies import AvellanedaStoikovPolicy, MLPolicy
from src.environments.util import ASState
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import time


def simulate_policy(env):
    action = env.policy.get_action()
    while True:
        obs, _, _, _ = env.step(action)
        action = env.policy.get_action()
        # print(env.get_total_value())


def simulate_ml(env, model):
    env.logging = True
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            break


def train_model(env):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20_000)
    # model.save(model_path)
    print("model trained")
    return model


def simulate_AS():
    ts = time.time()
    logging.basicConfig(
        filename=f"logs/as_env_sample_{int(ts)}.log",
        encoding="utf-8",
        level=logging.INFO,
    )
    env = MMEnv(
        state_folder="./parsed_data/AvellanedaStoikov/AS_full.csv",
        policy=AvellanedaStoikovPolicy,
        capital=1000,
        step_interval=10_000,
        price_decimals=4,
        ticks_size=0.0001,
        inventory_target=0,
        output_type=ASState,
        policy_params={"risk_aversion": 0.1, "order_size": 1},
        logging=True,
        logger=logging.getLogger(__name__),
    )
    simulate_policy(env)


def simulate_ml_as_param():
    ts = time.time()
    logging.basicConfig(
        filename=f"logs/as_env_sample_{int(ts)}.log",
        encoding="utf-8",
        level=logging.INFO,
    )
    env = MMEnv(
        state_folder="./parsed_data/AvellanedaStoikov/AS_full.csv",
        policy=MLPolicy,
        capital=1000,
        step_interval=10_000,
        price_decimals=4,
        tick_size=0.0001,
        inventory_target=0,
        output_type=ASState,
        policy_params={"max_size": 10},
        logging=False,
        logger=logging.getLogger(__name__),
    )
    model = train_model(env)
    simulate_ml(env, model)


if __name__ == "__main__":
    simulate_ml_as_param()
