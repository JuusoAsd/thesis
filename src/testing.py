import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import gym

from stable_baselines3 import PPO, A2C
import copy

from environments.env_configs.spaces import ActionSpace, ObservationSpace
from environments.env_configs.policies import ASPolicyVec
from environments.mm_env_vec import MMVecEnv, SBMMVecEnv

from environments.env_configs.rewards import *

load_dotenv()

print(f"reading data from: {os.getenv('BASE_PATH')}")
base = pd.read_csv(os.getenv("BASE_PATH"))
indicators = pd.read_csv(os.getenv("INDICATOR_PATH"))
data = pd.merge(base, indicators, on="timestamp", how="left").ffill().dropna()
data["mid_price"] = np.round((data.best_bid + data.best_ask) / 2, 5)
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
)
from environments.mm_env_vec import MMVecEnv, SBMMVecEnv
from environments.env_configs.util import ExponentialBetaSchedule, BCEvalCallback
from environments.env_configs.rewards import *
import environments.env_configs.policies as policies
import random


def run_random_initialized_model():
    # initialize a random ML model and see if every 10k action is constant
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
        reward_class=InventoryIntegralPenalty,
    )
    venv = SBMMVecEnv(env)
    model = PPO("MlpPolicy", venv, verbose=1, seed=random.randint(0, 1000))
    expert = expert_policy(env=env, **expert_params)
    action_func = expert.get_action_func()
    obs = venv.reset()

    done = False
    n_steps = 0
    print_step = 10_000
    while not done:
        action = action_func(obs)
        model_action = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        n_steps += 1
        if n_steps % print_step == 0:
            print(f"Random init model action; {model_action[0]}")


def run_random_initialized_model_new_model():
    # initialize a random ML model and see if every 10k action is constant
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
        reward_class=InventoryIntegralPenalty,
    )
    venv = SBMMVecEnv(env)
    model = A2C("MlpPolicy", venv, verbose=1, seed=random.randint(0, 1000))
    expert = expert_policy(env=env, **expert_params)
    action_func = expert.get_action_func()
    obs = venv.reset()

    done = False
    n_steps = 0
    print_step = 10_000
    while not done:
        action = action_func(obs)
        model_action = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        n_steps += 1
        if n_steps % print_step == 0:
            print(f"Random init model action; {model_action[0]}")


def run_random_initialized_model_non_vec():
    # initialize a random ML model and see if every 10k action is constant
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
        reward_class=InventoryIntegralPenalty,
    )
    model = PPO("MlpPolicy", env, verbose=1, seed=random.randint(0, 1000))
    expert = expert_policy(env=env, **expert_params)
    action_func = expert.get_action_func()
    obs = env.reset()

    done = False
    n_steps = 0
    print_step = 10_000
    while not done:
        action = action_func(obs)
        model_action = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        n_steps += 1
        if n_steps % print_step == 0:
            print(f"Random init non-vec model action: {model_action[0]}")

    for i in range(5):
        obs = env.observation_space.sample()
        action = model.predict(obs, deterministic=True)
        print(f"Random init non-vec model action sampled: {action[0]}")


def run_random_initialized_model_non_vec_no_size():
    # initialize a random ML model and see if every 10k action is constant
    obs_space = ObservationSpace.DummyObservation
    act_space = ActionSpace.NoSizeAction

    env = MMVecEnv(
        data.to_numpy(),
        n_envs=1,
        params={
            "observation_space": obs_space,
            "action_space": act_space,
        },
        column_mapping=column_mapping,
        reward_class=InventoryIntegralPenalty,
    )
    model = PPO("MlpPolicy", env, verbose=1, seed=random.randint(0, 1000))
    obs = env.reset()

    for i in range(10):
        obs = env.observation_space.sample()
        action = model.predict(obs, deterministic=True)
        print(f"Random init non-vec model action sampled: {action[0]}")


from cloning import save_trained_model, load_trained_model


def load_random_init_model():
    # Check if only a model that has been loaded from a file produces the constant output
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
        reward_class=InventoryIntegralPenalty,
    )
    venv = SBMMVecEnv(env)
    model = PPO("MlpPolicy", venv, verbose=1)
    expert = expert_policy(env=env, **expert_params)
    action_func = expert.get_action_func()

    # save the model
    save_trained_model(model, "random_init_model")
    loaded_model = load_trained_model("random_init_model", venv, normalize=False)
    done = False
    all_val = []
    n_steps = 0
    print_step = 10_000
    obs = venv.reset()

    while not done:
        action = action_func(obs)
        model_action = loaded_model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        current_val = obs[0].tolist() + action[0].tolist()
        all_val.append(current_val)
        n_steps += 1
        if n_steps % print_step == 0:
            print(f"Loaded model action: {model_action}")


from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv


class SimpleEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

    def step(self, action):
        rand = random.randint(0, 1_000)
        if rand > 995:
            done = True
        else:
            done = False
        return self._get_obs(), 0, done, {}

    def reset(self):
        return self._get_obs()

    def _get_obs(self):
        obs = random.uniform(-1, 1)
        self.previous_obs = obs
        return obs


def get_round_func():
    def round_func(x):
        return np.round(x, 2)

    return round_func


def test_simple_policy():
    env = SimpleEnv()
    model = PPO("MlpPolicy", env, verbose=1, seed=random.randint(0, 1000))
    for i in range(10):
        obs = env.observation_space.sample()
        action = model.predict(obs, deterministic=True)
        print(f"Random init simple env action sampled: {action[0]}")


def test_simple_policy_vec():
    env = SimpleEnv()
    env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
    model = PPO("MlpPolicy", env, verbose=1, seed=random.randint(0, 1000))
    for i in range(10):
        obs = env.observation_space.sample()
        action = model.predict(obs, deterministic=True)
        print(f"Random init simple env vec action sampled: {action[0]}")


def test_imitation_simple_policy():
    env = SimpleEnv()
    model = PPO("MlpPolicy", env, verbose=1, seed=random.randint(0, 1000))
    rng = (np.random.default_rng(15),)

    rollouts = rollout.rollout(
        get_round_func(),
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=1000),
        rng=rng,
        unwrap=False,
    )
    transitions = rollout.flatten_trajectories(rollouts)


from environments.env_configs.spaces import LinearObservation, LinearObservationSpaces


def test_linear_obs():
    obs_space = LinearObservation(LinearObservationSpaces.SimpleSpace)
    # print(obs_space.mapping_to_normalized)
    obs = {"inventory": 0.5}
    res = obs_space.convert_to_normalized(obs_dict=obs)
    # print(res)

    # print(obs_space.convert_to_normalized({"inventory": 100_000}))
    print(obs_space.convert_to_normalized({"inventory": 0.5}))
    print(obs_space.convert_to_normalized({"volatility": 0.5}))
    print(obs_space.convert_to_normalized({"intensity": 0.5}))


def test_no_size_normalized():
    # First setup an environment where we use normalized actions and step through it
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

    env_normalized = MMVecEnv(
        data.to_numpy(),
        n_envs=1,
        params={
            "observation_space": obs_space,
            "action_space": act_space,
        },
        column_mapping=column_mapping,
        reward_class=InventoryIntegralPenalty,
    )
    expert_normalized = expert_policy(env=env_normalized, **expert_params)
    action_func = expert_normalized.get_action_func()
    obs = env_normalized.reset()

    done = False
    n_steps = 0
    print_step = 10_000
    while not done:
        action = action_func(obs)
        # action_no_size = action_func_expert_no_size(obs)
        obs, reward, done, info = env_normalized.step(action)
        n_steps += 1
        if n_steps % print_step == 0:
            print(f"Obs: {obs}")
            print(
                f"State: {env_normalized.bid_sizes}, {env_normalized.ask_sizes}, {env_normalized.bids}, {env_normalized.asks} "
            )
            print(f"Random init non-vec model action: {action}")

    # then setup an environment where we use no size actions and step through it
    expert_params["act_type"] = ActionSpace.NoSizeAction
    act_space = ActionSpace.NoSizeAction
    expert_no_size = ASPolicyVec(env_normalized, **expert_params)
    env_no_size = MMVecEnv(
        data.to_numpy(),
        n_envs=1,
        params={
            "observation_space": obs_space,
            "action_space": act_space,
        },
        column_mapping=column_mapping,
        reward_class=InventoryIntegralPenalty,
    )
    print("change: \n")
    action_func_expert_no_size = expert_no_size.get_action_func()

    done = False
    n_steps = 0
    obs = env_no_size.reset()
    while not done:
        action_no_size = action_func_expert_no_size(obs)
        obs, reward, done, info = env_no_size.step(action_no_size)
        n_steps += 1
        if n_steps % print_step == 0:
            print(f"Obs: {obs}")
            print(
                f"State: {env_no_size.bid_sizes}, {env_no_size.ask_sizes}, {env_no_size.bids}, {env_no_size.asks} "
            )
            print(f"Random init non-vec model action: {action_no_size}")


def test_linear_obs():
    obs_space = LinearObservation(LinearObservationSpaces.SimpleSpace)
    # print(obs_space.mapping_to_normalized)
    obs = {"inventory": 0.5}
    res = obs_space.convert_to_normalized(obs_dict=obs)
    # print(res)

    # print(obs_space.convert_to_normalized({"inventory": 100_000}))
    print(obs_space.convert_to_normalized({"inventory": 0.5}))
    print(obs_space.convert_to_normalized({"volatility": 0.5}))
    print(obs_space.convert_to_normalized({"intensity": 0.5}))


def test_overlapping():
    # First setup an environment where we use normalized actions and step through it
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
        "obs_type": obs_space,
        "act_type": act_space,
    }

    env_normalized = MMVecEnv(
        data.to_numpy(),
        n_envs=1,
        params={
            "observation_space": obs_space,
            "action_space": act_space,
        },
        column_mapping=column_mapping,
        reward_class=InventoryIntegralPenalty,
    )
    expert_normalized = expert_policy(env=env_normalized, **expert_params)
    action_func = expert_normalized.get_action_func()

    expert_params["act_type"] = ActionSpace.NoSizeAction
    act_space = ActionSpace.NoSizeAction
    expert_no_size = ASPolicyVec(env_normalized, **expert_params)
    env_no_size = MMVecEnv(
        data.to_numpy(),
        n_envs=1,
        params={
            "observation_space": obs_space,
            "action_space": act_space,
        },
        column_mapping=column_mapping,
        reward_class=InventoryIntegralPenalty,
    )
    action_func_expert_no_size = expert_no_size.get_action_func()

    done = False
    n_steps = 0
    print_step = 10_000
    obs_normalized = env_normalized.reset()
    obs_no_size = env_no_size.reset()
    steps = 100_000
    while not done:
        action_normalized = action_func(obs_normalized)
        obs_normalized, reward, done, info = env_normalized.step(action_normalized)
        action_no_size = action_func_expert_no_size(obs_no_size)
        obs_no_size, reward, done, info = env_no_size.step(action_no_size)

        n_steps += 1
        if n_steps % print_step == 0:
            print(
                f"State: {env_normalized.bid_sizes}, {env_normalized.ask_sizes}, {env_normalized.bids}, {env_normalized.asks} "
            )
            print(
                f"State: {env_no_size.bid_sizes}, {env_no_size.ask_sizes}, {env_no_size.bids}, {env_no_size.asks}\n"
            )
        if n_steps >= steps:
            break


from environments.util import setup_venv
from data_management import get_data_by_dates
from cloning import load_trained_model
import time


def test_step_speed():
    data = get_data_by_dates("2021-12-23")
    venv = setup_venv(data=data, n_env=8)
    model = load_trained_model("clone_bc", venv)
    obs_model = venv.reset()
    done = False
    n_steps = 0
    print_step = 10_000
    start_time = time.time()
    while not done:
        action_model = model.predict(obs_model, deterministic=True)[0]
        obs_model, _, dones, _ = venv.step(action_model)
        done = np.all(dones)

        n_steps += 1
    end_time = time.time()
    total_time = end_time - start_time
    time_per_step = total_time / n_steps
    time_per_step_per_env = time_per_step / venv.env.n_envs
    print(f"using {venv.env.n_envs} envs: {time_per_step_per_env} per step per env")

    data = get_data_by_dates("2021-12-23")
    venv = setup_venv(data=data, n_env=32)
    model = load_trained_model("clone_bc", venv)
    obs_model = venv.reset()
    done = False
    n_steps = 0
    print_step = 10_000
    start_time = time.time()
    while not done:
        action_model = model.predict(obs_model, deterministic=True)[0]
        obs_model, _, dones, _ = venv.step(action_model)
        done = np.all(dones)
        n_steps += 1
    end_time = time.time()
    total_time = end_time - start_time
    time_per_step = total_time / n_steps
    time_per_step_per_env = time_per_step / venv.env.n_envs
    print(f"using {venv.env.n_envs} envs: {time_per_step_per_env} per step per env")


def test_cloning_vs_manual():
    data = get_data_by_dates("2021_12_23")
    venv = setup_venv(
        data=data,
        act_space=ActionSpace.NormalizedAction,
    )
    model = load_trained_model("clone_bc", venv)

    expert_venv = copy.deepcopy(venv)

    expert_policy = ASPolicyVec
    expert_params = {
        "max_order_size": 5,
        "tick_size": 0.0001,
        "max_ticks": 10,
        "price_decimals": 4,
        "inventory_target": 0,
        "risk_aversion": 0.2,
        "order_size": 1,
        "obs_type": venv.env.obs_space,
        "act_type": venv.env.act_space,
    }

    expert = expert_policy(env=expert_venv.env, **expert_params)
    action_func = expert.get_action_func()
    obs_model = venv.reset()
    obs_expert = expert_venv.reset()
    done = False
    n_steps = 0
    print_step = 10_000
    while not done:
        action_model = model.predict(obs_model, deterministic=True)[0]
        action_expert = action_func(obs_expert)

        obs_model, _, done, _ = venv.step(action_model)
        obs_expert, _, done, _ = expert_venv.step(action_expert)

        n_steps += 1

    print(f"Model: {venv.env.get_metrics()}")
    print(f"Expert: {expert_venv.env.get_metrics()}")


def test_inventory_environments():
    data = get_data_by_dates("2021-12-23")
    venv = setup_venv(
        data=data, time_envs=1, inv_envs=10, env_params={"inv_jump": 0.09}
    )
    print(venv.reset())
    print(venv.env._get_value())


def test_trained_model():
    model_date = "2021_12_23"
    data_date = "2021_12_24"
    data = get_data_by_dates(data_date)
    venv = setup_venv(
        data=data,
        act_space=ActionSpace.NormalizedAction,
    )
    expert_venv = copy.deepcopy(venv)
    model = load_trained_model(model_date, venv)
    expert_policy = ASPolicyVec
    expert_params = {
        "max_order_size": 5,
        "tick_size": 0.0001,
        "max_ticks": 10,
        "price_decimals": 4,
        "inventory_target": 0,
        "risk_aversion": 0.2,
        "order_size": 1,
        "obs_type": venv.env.obs_space,
        "act_type": venv.env.act_space,
    }

    expert = expert_policy(env=expert_venv.env, **expert_params)
    action_func = expert.get_action_func()
    obs_model = venv.reset()
    obs_expert = expert_venv.reset()
    done = False
    n_steps = 0
    print_obs = 10_000
    while not done:
        action_model = model.predict(obs_model, deterministic=True)[0]
        action_expert = action_func(obs_expert)

        obs_model, _, done, _ = venv.step(action_model)
        obs_expert, _, done, _ = expert_venv.step(action_expert)

        n_steps += 1

        if n_steps % print_obs == 0:
            print(f"Model: {action_model.tolist()}")
            print(f"Expert: {action_expert.tolist()}")
    print(f"Model: {venv.env.get_metrics()}")
    print(f"Expert: {expert_venv.env.get_metrics()}")


if __name__ == "__main__":
    # run_random_initialized_model()
    # run_random_initialized_model_non_vec()
    # run_random_initialized_model_new_model()
    # test_simple_policy()
    # test_simple_policy_vec()
    # run_random_initialized_model_non_vec_no_size()
    # test_imitation_simple_policy()
    # load_random_init_model()
    # test_linear_obs()
    # test_no_size_normalized()
    # test_overlapping()
    # test_cloning_vs_manual()

    # test_step_speed()
    # test_inventory_environments()
    test_trained_model()
