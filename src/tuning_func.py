import os
from datetime import datetime, timedelta
import logging
import time

import pandas as pd
from ray.tune.search.repeater import TRIAL_INDEX
from stable_baselines3 import PPO
from omegaconf import OmegaConf


# from testing import test_trained_vs_manual
from src.cloning import (
    load_trained_model,
    load_model_by_config,
)
from src.environments.util import setup_venv, setup_venv_config
from src.data_management import get_data_by_dates
from src.util import (
    set_seeds,
    get_config,
    de_flatten_config,
    locked_write_dataframe_to_csv,
    create_config_hash,
)
from src.environments.env_configs.spaces import ActionSpace
from src.environments.env_configs.rewards import AssymetricPnLDampening
from src.environments.env_configs.callbacks import ExternalMeasureCallback


os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"


def objective_simple(
    config,
):
    # multiple objectives are run by repeater, set seed based on trial index
    set_seeds(config[TRIAL_INDEX])
    load_model_name = "clone_bc"
    start_date = datetime.strptime(common_config.objective.tune_train_start, "%Y_%m_%d")
    duration = common_config.objective.tune_train_duration
    end_date = start_date + timedelta(days=duration)
    train_data = get_data_by_dates(start_date, end_date)
    reward = AssymetricPnLDampening
    base_venv = setup_venv(
        data=train_data,
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

    validation_data = get_data_by_dates(
        end_date, end_date + timedelta(days=common_config.objective.validation_duration)
    )
    measure_callback = ExternalMeasureCallback(
        data=validation_data.to_numpy(),
        venv=base_venv,
        model_name=f"cloned_{reward.__name__}_init",
        **common_config.objective.validation,
    )
    model = load_trained_model(
        load_model_name,
        base_venv,
        model_kwargs=config,
    )
    model.learn(
        total_timesteps=common_config.objective.timesteps,
        log_interval=None,
        callback=measure_callback,
    )

    if measure_callback.best_performance_metrics == {}:
        measure_callback._on_rollout_end()
    return_dict = {"trial_reward": measure_callback.best_reward, "group_reward": 0}
    return_dict.update(measure_callback.best_performance_metrics)

    return return_dict
    # return {"trial_reward": random.randint(1, 2), "lolno": 1, "group_reward": 0}


def violate_constraints(config):
    """
    check env specific constrains.
    - mini-batch size should be a factor of n_steps * n_envs
    """

    env_size = config.env.params.inv_envs * config.env.params.time_envs
    n_steps = config.model.model_params.n_steps
    buffer_size = n_steps * env_size
    mini_batch = config.model.model_params.batch_size

    if buffer_size % mini_batch > 0:
        return True
    else:
        return False


def objective_clone(config_dict):
    """
    Objective function that
    - clones a model using BC and saves the cloned model
    - repeats following process n times:
        - loads the cloned model using different seed
        - trains the loaded model on the training data
        - evaluates the model on the validation data
        - returns the evaluation metrics
    - eventually returns an aggregate of the evaluation metrics
    """
    start_time = time.time()
    un_flat = de_flatten_config(config_dict)
    config = OmegaConf.create(un_flat)

    # double callback wait time if not cloning
    config.tuning.callback.wait = (
        config.tuning.callback.wait * 2
        if not config.clone
        else config.tuning.callback.wait
    )

    # if violate_constraints(config):
    #     print("violated constraints, interrupting")
    #     return {"agg_reward": -1, "sharpe": -1, "mean_abs_inv": -1, "returns": -1}

    seed = 0
    set_seeds(seed)

    # for given config, setup the environment
    venv = setup_venv_config(config.train_data, config.env, config.venv)

    # setup the model
    algo_dict = {"PPO": PPO}

    callback_data = get_data_by_dates(**config.eval_data)
    # get results from training the model
    trial_unique_id = create_config_hash(config)
    rewards = []
    for i in range(config.tuning.repeats):
        logging.info(f"Running trial {i}")
        if config.clone:
            model = load_model_by_config(config, venv)
        else:
            model = algo_dict[config.model.algo](
                policy=config.model.policy,
                env=venv,
                **config.model.model_params,
                policy_kwargs=config.model.policy_kwargs,
            )
        tune_venv = setup_venv_config(config.eval_data, config.env, config.venv)
        set_seeds(i)
        callback = ExternalMeasureCallback(
            data=callback_data.to_numpy(), venv=tune_venv, **config.tuning.callback
        )
        model.learn(
            total_timesteps=config.tuning.timesteps,
            log_interval=None,
            callback=callback,
        )
        # for each trial, save the average of best performance metrics
        metrics = callback.best_performance_metrics
        metrics["trial_nr"] = i
        metrics["trial_id"] = trial_unique_id
        metrics["best_reward"] = callback.best_reward

        rewards.append(metrics)

    reward_df = pd.DataFrame(rewards)
    agg_reward = reward_df.best_reward.mean() / (reward_df.best_reward.std() + 1e-6)
    sharpe = reward_df.sharpe.mean()
    mean_abs_inv = reward_df.mean_abs_inv.mean()
    returns = reward_df.episode_return.mean()
    duration = round((time.time() - start_time) / 60, 2)

    locked_write_dataframe_to_csv(config.run_name, "rewards", reward_df)

    config_dict["trial_id"] = trial_unique_id
    config_dict["reward"] = agg_reward
    config_dict["sharpe"] = sharpe
    config_dict["mean_abs_inv"] = mean_abs_inv
    config_dict["episode_return"] = returns
    config_dict["duration"] = duration
    config_df = pd.DataFrame([config_dict])
    locked_write_dataframe_to_csv(config.run_name, "parameters", config_df)
    return {
        "agg_reward": agg_reward,
        "sharpe": sharpe,
        "mean_abs_inv": mean_abs_inv,
        "returns": returns,
        "duration": duration,
    }


def objective_preload_repeat(config_dict):
    """
    Objective function where:
    - model is initialized with a cloned model (or not depending on config)
    - there are multiple architecture options for the model
    - repeats of the trials are performed using repeater rather than loop in objective
    """
    start_time = time.time()
    un_flat = de_flatten_config(config_dict)
    config = OmegaConf.create(un_flat)
    set_seeds(config[TRIAL_INDEX])
    config.pop(TRIAL_INDEX)
    run_hash = create_config_hash(config)

    # for given config, setup the environment
    venv = setup_venv_config(config.train_data, config.env, config.venv)

    # setup the model
    algo_dict = {"PPO": PPO}

    eval_data = get_data_by_dates(**config.eval_data)
    if config.clone:
        model_hash = create_config_hash(config.model.policy_kwargs)
        config.model_name = model_hash
        model = load_model_by_config(config, venv)
    else:
        kwarg_dict = OmegaConf.to_container(config.model.policy_kwargs, resolve=True)
        model = algo_dict[config.model.algo](
            policy=config.model.policy,
            env=venv,
            **config.model.model_params,
            policy_kwargs=kwarg_dict,
        )
    tune_venv = setup_venv_config(config.eval_data, config.env, config.venv)
    callback = ExternalMeasureCallback(
        data=eval_data.to_numpy(), venv=tune_venv, **config.tuning.callback
    )
    model.learn(
        total_timesteps=config.tuning.timesteps,
        log_interval=None,
        callback=callback,
    )
    metrics = callback.best_performance_metrics
    metrics["trial_reward"] = callback.best_reward
    duration = round((time.time() - start_time) / 60, 2)
    metrics["duration"] = duration
    metrics["trial_group"] = run_hash
    reward_df = pd.DataFrame([metrics])

    locked_write_dataframe_to_csv(config.run_name, "trial_results", reward_df)
    config_dict["trial_group"] = run_hash
    config_df = pd.DataFrame([config_dict])

    locked_write_dataframe_to_csv(config.run_name, "trial_parameters", config_df)
    return {
        "trial_group": run_hash,
        "trial_reward": metrics["trial_reward"],
        "sharpe": metrics["sharpe"],
        "returns": metrics["episode_return"],
        "max_inventory": metrics["max_inventory"],
        "mean_abs_inv": metrics["mean_abs_inv"],
        "duration": duration,
    }


if __name__ == "__main__":
    # config = get_config("tuning_train_eval_single_run")
    # print(config)
    # objective_clone(config)

    config = get_config("tuning_test_objective_preload_repeat")
    objective_preload_repeat(config)
