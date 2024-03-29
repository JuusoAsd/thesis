import os
import logging
import copy
import csv
import pandas as pd

import numpy as np
from dotenv import load_dotenv


from src.environments.env_configs.spaces import ActionSpace
from src.environments.env_configs.policies import ASPolicyVec
from src.environments.env_configs.rewards import *
from src.data_management import get_data_by_dates
from src.environments.env_configs.spaces import get_action_space_keys

load_dotenv()


def test_cloning_vs_manual():
    from src.environments.util import setup_venv
    from src.cloning import load_trained_model

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


def test_trained_model(venv, model, recorded_metrics="summary"):
    n_steps = 0

    action_list = []
    obs_list = []
    done = False
    obs = venv.reset()
    while not done:
        action = model.predict(obs, deterministic=True)[0]
        if recorded_metrics == "full":
            obs_list.append(obs[0].tolist())
            action_list.append(action[0].tolist())
        obs, _, done, _ = venv.step(action)
        n_steps += 1
    if recorded_metrics == "summary":
        if venv.env.n_envs == 1:
            return venv.env.get_metrics_single()
        else:
            return venv.env.get_metrics()
    elif recorded_metrics == "full":
        action_df = pd.DataFrame(
            action_list, columns=get_action_space_keys(venv.env.act_space)
        )
        obs_df = pd.DataFrame(
            obs_list, columns=list(venv.env.obs_space.obs_info_dict.keys())
        )
        df = venv.env.get_recorded_values_to_df()
        df["mid_price"] = venv.env.mid_price
        df["best_bid"] = venv.env.best_bid
        df["best_ask"] = venv.env.best_ask
        df["low_price"] = venv.env.low_price
        df["high_price"] = venv.env.high_price
        df["timestamp"] = venv.env.timestamp
        df = pd.concat([df, action_df, obs_df], axis=1)
        return df


def trained_vs_manual(venv, model, save_values=False, result_file="", date=""):
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
    }

    expert = expert_policy(env=expert_venv.env, **expert_params)

    action_func = expert.get_action_func()
    obs_model = venv.reset()
    obs_expert = expert_venv.reset()
    n_steps = 0
    venv.env.reset_metrics_on_reset = False
    expert_venv.env.reset_metrics_on_reset = False
    venv.reset()
    while True:
        action_model = model.predict(obs_model, deterministic=True)[0]
        action_expert = action_func(obs_expert)
        obs_model, _, _, _ = venv.step(action_model)
        obs_expert, _, done_expert, _ = expert_venv.step(action_expert)
        n_steps += 1
        if done_expert:
            break

    # metrics = venv.env.get_metrics()
    # if metrics["max_inventory"] == 0:
    #     logging.error("Zero inventory run")
    #     return
    model_metrics = venv.env.get_metrics_val()
    expert_metrics = expert_venv.env.get_metrics_val()
    if save_values:
        if date == "":
            raise ValueError("Date must be specified when saving")
        date_str = date.strftime("%Y_%m_%d")
        model_metrics["date"] = date_str
        expert_metrics["date"] = date_str
        model_metrics["result_type"] = "model"
        expert_metrics["result_type"] = "expert"
        path = os.path.join(os.getenv("RESULT_PATH"), f"{result_file}.csv")
        with open(path, "a+") as f:
            w = csv.DictWriter(f, model_metrics.keys())
            w.writerow(model_metrics)
            w.writerow(expert_metrics)

    return (model_metrics, expert_metrics)


def test_callback():
    dates = "2021_12_30"
    data = get_data_by_dates(dates)
    reward = InventoryIntegralPenalty

    venv = setup_venv(
        data=data,
        act_space=ActionSpace.NormalizedAction,
        reward_class=InventoryIntegralPenalty,
        inv_envs=1,
        time_envs=1,
        env_params={"inv_jump": 0.18, "data_portion": 0.5},
    )
    model = load_trained_model(
        # f"clone_to_{reward.__name__}_{start_str}_to_{end_str}",
        # "cloned_AssymetricPnLDampening_best",
        "clone_bc",
        venv,
    )
    callback = ExternalMeasureCallback(
        data=data.to_numpy(), venv=venv, wait=0, freq=1, time_envs=2
    )
    model.learn(total_timesteps=3000, callback=callback)
    print(
        f"metrics: {callback.best_performance_metrics} \n best: {callback.best_reward}"
    )


if __name__ == "__main__":
    from src.cloning import load_trained_model
    from src.environments.util import setup_venv_config
    from src.util import get_config

    config = get_config("test_config")

    venv = setup_venv_config(config.data, config.env, config.venv)
    model = load_trained_model("clone_large", venv, normalize=False)
    test_trained_vs_manual(
        venv,
        model,
    )
