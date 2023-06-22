import os
import pandas as pd
from datetime import datetime, timedelta
from stable_baselines3 import PPO

from src.data_management import get_data_by_dates, get_data_by_date_list
from src.cloning import load_trained_model, save_model, load_model
from src.util import get_config, flatten_config, de_flatten_config
from src.environments.util import setup_venv
from src.environments.env_configs.spaces import ActionSpace
from src.environments.env_configs.rewards import (
    PnLReward,
    AssymetricPnLDampening,
    InventoryIntegralPenalty,
    SpreadPnlReward,
)
from src.environments.env_configs.callbacks import ExternalMeasureCallback
from src.util import create_config_hash
from src.cloning import setup_venv_config, load_model_by_config, save_model_by_config
from omegaconf import OmegaConf
import time
import logging
from src.model_testing import test_trained_model
from src.environments.env_configs.policies import ASPolicyVec


def create_date_intervals(start_date, end_date, interval, skipped_dates):
    all_dates = [
        start_date + timedelta(days=i)
        for i in range((end_date - start_date).days + 1)
        if start_date + timedelta(days=i) not in skipped_dates
    ]
    if interval == 0:
        return all_dates
    date_ranges = [
        all_dates[i : i + interval] for i in range(0, len(all_dates), interval)
    ]
    return date_ranges


def rolling_test_train():
    logging.basicConfig(level=logging.INFO)
    base_config = get_config("base_rolling_train_test")
    config = get_config(base_config.config_name, subdirectory=["good_configs"])
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, base_config)
    start_time = time.time()
    run_hash = create_config_hash(config)

    logging.info(f"Run name: {config.run_name}")
    logging.info(f"Init cloned: {config.clone}")
    logging.info(f"Retrain: {config.retrain}")
    logging.info(
        f"Initial train data: {config.initial_train.train_data.start_date} - {config.initial_train.train_data.end_date}"
    )
    logging.info(
        f"Test data: {config.rolling_test_train.start_date} - {config.rolling_test_train.end_date} with interval {config.rolling_test_train.retrain_interval}"
    )

    # for given config, setup the environment
    venv = setup_venv_config(config.initial_train.train_data, config.env, config.venv)

    # setup the model
    algo_dict = {"PPO": PPO}

    eval_data = get_data_by_dates(**config.initial_train.eval_data)

    # check if model is already trained
    model_path = f"{os.getenv('COMMON_PATH')}/models/{config.run_name}_{config.config_name}_init.zip"
    if os.path.exists(model_path) and not config.retrain:
        # logging.info("Model already trained, loading model")
        tune_venv = setup_venv_config(config.eval_data, config.env, config.venv)
        model = load_model(
            f"{config.run_name}_{config.config_name}_init", tune_venv, ["models"]
        )
    else:
        if config.clone:
            hash_config = OmegaConf.create(
                {
                    "policy_kwargs": config.model.policy_kwargs,
                    "action": config.env.spaces.action_space,
                    "observation": config.env.spaces.observation_space.params,
                }
            )
            model_hash = create_config_hash(hash_config)
            config.model["model_name"] = model_hash
            model = load_model_by_config(config, venv)
        else:
            kwarg_dict = OmegaConf.to_container(
                config.model.policy_kwargs, resolve=True
            )
            model = algo_dict[config.model.algo](
                policy=config.model.policy,
                env=venv,
                **config.model.model_params,
                policy_kwargs=kwarg_dict,
            )
        tune_venv = setup_venv_config(config.eval_data, config.env, config.venv)
        callback = ExternalMeasureCallback(
            data=eval_data.to_numpy(),
            venv=tune_venv,
            **config.initial_train.callback,
            save_best_model=True,
            model_name=f"{config.run_name}_{config.config_name}_init",
        )
        logging.info(f"Starting initial learning")
        model.learn(
            total_timesteps=config.initial_train.timesteps,
            log_interval=None,
            callback=callback,
        )
        if not callback.has_saved:
            logging.error(f"Training did not find a better performance, continuing")
            # save model here so show goes on
            save_model(
                model, f"{config.run_name}_{config.config_name}_init", ["models"]
            )

    model = load_model(
        f"{config.run_name}_{config.config_name}_init", tune_venv, ["models"]
    )

    """
    at this point model should be trained on the initial data
    now move to new data, first testing on the batch, then training on it
    this loop is then repeated ad infinitum
    """

    start_date = datetime.strptime(config.rolling_test_train.start_date, "%Y_%m_%d")
    end_date = datetime.strptime(config.rolling_test_train.end_date, "%Y_%m_%d")
    interval = config.rolling_test_train.retrain_interval
    skipped = config.rolling_test_train.skipped_days
    skipped_dates = [datetime.strptime(date, "%Y_%m_%d") for date in skipped]

    date_ranges = create_date_intervals(start_date, end_date, interval, skipped_dates)
    eval_data = get_data_by_dates(**config.rolling_test_train.eval_data)

    # save the model placeholder so can load it in different environment for testing
    save_model(model, f"{config.run_name}_ph", ["models"])
    result_list = []
    for range in date_ranges:
        if interval == 0:
            logging.info(f"Testing on full range")
            range_data = get_data_by_date_list(date_ranges)
        else:
            logging.info(f"Testing on {range[0]}")
            range_data = get_data_by_date_list(range)

        # start by testing performance of the current model
        test_venv = venv.clone_venv(data=range_data.to_numpy())
        test_model = load_model(f"{config.run_name}_ph", test_venv, ["models"])
        test_venv.env.reset_metrics_on_reset = False
        metrics = test_trained_model(
            test_venv, test_model, recorded_metrics=config.recorded_metrics
        )
        if interval != 0:
            if len(range) != 1:
                metrics["start_date"] = range[0]
                metrics["end_date"] = range[-1]
            else:
                metrics["date"] = range[0]
            result_list.append(metrics)
        else:
            result_list.append(metrics)
            break
        if config.rolling_test_train.timesteps != 0:
            # now train the model on the same data
            callback = ExternalMeasureCallback(
                data=eval_data.to_numpy(),
                venv=tune_venv,
                **config.rolling_test_train.callback,
                model_name=f"{config.run_name}_ph",
            )

            model.learn(
                total_timesteps=config.rolling_test_train.timesteps,
                log_interval=None,
                callback=callback,
            )
            if not callback.has_saved:
                # first evaluation of model is always actually saved
                logging.error(
                    f"Training did not find a better performance for {range}, continuing"
                )
    if config.recorded_metrics == "summary":
        df = pd.DataFrame(result_list)
    elif config.recorded_metrics == "full":
        df = pd.DataFrame()
        for i in result_list:
            df = pd.concat([df, i])

    path = os.path.join(
        os.getenv("RESULT_PATH"),
        "testing",
        f"{config.run_name}_{config.recorded_metrics}_{config.name_id}.csv",
    )
    df.to_csv(path, index=False)
    print(f"Saved at: {path}")


def rolling_train_test_as():
    logging.basicConfig(level=logging.INFO)
    base_config = get_config("base_rolling_train_test")
    config = get_config(base_config.config_name, subdirectory=["good_configs"])
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, base_config)
    OmegaConf.set_struct(config, False)
    del config["env"]["params"]["inv_envs"]
    del config["env"]["params"]["time_envs"]
    del config["env"]["params"]["data_portion"]
    del config["env"]["params"]["inv_jump"]

    venv = setup_venv_config(config.initial_train.train_data, config.env, config.venv)

    interval = config.rolling_test_train.retrain_interval
    skipped = config.rolling_test_train.skipped_days
    skipped_dates = [datetime.strptime(date, "%Y_%m_%d") for date in skipped]
    start_date = datetime.strptime(config.rolling_test_train.start_date, "%Y_%m_%d")
    end_date = datetime.strptime(config.rolling_test_train.end_date, "%Y_%m_%d")
    date_ranges = create_date_intervals(start_date, end_date, interval, skipped_dates)
    result_list = []
    if config.as_comparison.include_model:
        model = load_model(
            f"{config.run_name}_{config.config_name}_init", venv, ["models"]
        )
    # action_dict = {k: [] for k in config.as_comparison.actions}
    action_list = []
    obs_list = []
    for range in date_ranges:
        if interval == 0:
            logging.info(f"Testing on full range")
            range_data = get_data_by_date_list(date_ranges)
        else:
            logging.info(f"Testing on {range[0]}")
            range_data = get_data_by_date_list(range)

        test_venv = venv.clone_venv(data=range_data.to_numpy())
        test_venv.env.reset_metrics_on_reset = False
        expert = ASPolicyVec(env=test_venv.env, **config.expert_params)
        action_func = expert.get_action_func()
        obs = test_venv.reset()
        while True:
            action = action_func(obs)
            obs_list.append(obs[0].tolist())
            if config.as_comparison.include_model:
                model_action = model.predict(obs, deterministic=True)[0]

            obs, reward, done, info = test_venv.step(action)

            if config.as_comparison.save_full:
                actions = model_action[0].tolist() + action[0].tolist()
                action_list.append(actions)
            if done:
                break

        if config.recorded_metrics == "summary":
            if test_venv.env.n_envs == 1:
                metrics = test_venv.env.get_metrics_single()
            else:
                metrics = test_venv.env.get_metrics()
        elif config.recorded_metrics == "full":
            metrics = test_venv.env.get_recorded_values_to_df()
        if interval != 0:
            if len(range) != 1:
                metrics["start_date"] = range[0]
                metrics["end_date"] = range[-1]
            else:
                metrics["date"] = range[0]
            result_list.append(metrics)
        else:
            result_list.append(metrics)
            break

    columns_model = [f"{act}_model" for act in config.as_comparison.actions]
    columns_expert = [f"{act}_expert" for act in config.as_comparison.actions]
    action_df = pd.DataFrame(action_list, columns=columns_model + columns_expert)
    obs_df = pd.DataFrame(obs_list, columns=config.as_comparison.observations)

    if config.recorded_metrics == "summary":
        df = pd.DataFrame(result_list)
    elif config.recorded_metrics == "full":
        df = pd.DataFrame()
        for i in result_list:
            df = pd.concat([df, i])

        print(df.shape)
        print(venv.env.mid_price.shape)
        df["mid_price"] = venv.env.mid_price
        df["best_bid"] = venv.env.best_bid
        df["best_ask"] = venv.env.best_ask
        df["low_price"] = venv.env.low_price
        df["high_price"] = venv.env.high_price
        df["timestamp"] = venv.env.timestamp
        df = pd.concat([df, action_df, obs_df], axis=1)

    file = os.path.join(
        os.getenv("RESULT_PATH"),
        "testing",
        f"{config.run_name}_{config.recorded_metrics}_{config.name_id}_as_comparison={config.as_comparison.include_model}.csv",
    )
    print(f"Saving to: {file}")
    df.to_csv(
        file,
        index=False,
    )
