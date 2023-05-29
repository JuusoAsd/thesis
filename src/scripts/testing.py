import os
from datetime import datetime, timedelta
from stable_baselines3 import PPO

from src.data_management import get_data_by_dates, get_data_by_date_list
from src.cloning import load_trained_model, save_model, load_model
from src.util import get_config
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


def create_date_intervals(start_date, end_date, interval, skipped_dates):
    all_dates = [
        start_date + timedelta(days=i)
        for i in range((end_date - start_date).days + 1)
        if start_date + timedelta(days=i) not in skipped_dates
    ]
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

    # for given config, setup the environment
    venv = setup_venv_config(config.initial_train.train_data, config.env, config.venv)

    # setup the model
    algo_dict = {"PPO": PPO}

    eval_data = get_data_by_dates(**config.initial_train.eval_data)

    # check if model is already trained
    model_path = f"{os.getenv('COMMON_PATH')}/models/{config.run_name}_{config.config_name}_init.zip"
    if os.path.exists(model_path) and not config.retrain:
        logging.info("Model already trained, loading model")
        tune_venv = setup_venv_config(config.eval_data, config.env, config.venv)
        model = load_model(
            f"{config.run_name}_{config.config_name}_init", tune_venv, ["models"]
        )
    else:
        if config.clone:
            model_hash = create_config_hash(config.model.policy_kwargs)
            config.model.model_name = model_hash
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
    for range in date_ranges:
        range_data = get_data_by_date_list(range)

        # start by testing performance of the current model
        test_venv = venv.clone_venv(data=range_data.to_numpy())
        test_model = load_model(f"{config.run_name}_ph", test_venv, ["models"])
        test_venv.env.reset_metrics_on_reset = False
        logging.info(f"Testing on {range[0]}")
        test_trained_model(test_venv, test_model)

        # now train the model on the same data
        callback = ExternalMeasureCallback(
            data=eval_data.to_numpy(),
            venv=tune_venv,
            **config.rolling_test_train.callback,
            model_name=f"{config.run_name}_ph",
        )

        # model.learn(
        #     total_timesteps=config.rolling_test_train.timesteps,
        #     log_interval=None,
        #     callback=callback,
        # )
        if not callback.has_saved:
            # first evaluation of model is always actually saved
            logging.error(
                f"Training did not find a better performance for {range}, continuing"
            )
