import os
import gym
import numpy as np
from functools import partial
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air.config import RunConfig
from ray.tune.search.repeater import Repeater, TRIAL_INDEX
from ray.tune.search.hyperopt import HyperOptSearch

from datetime import datetime, timedelta
from stable_baselines3 import PPO
from testing import test_trained_model, test_trained_vs_manual
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
from environments.env_configs.callbacks import (
    ExternalMeasureCallback,
    GroupRewardCallback,
)
from ray.tune.search.bayesopt import BayesOptSearch
from util import set_seeds, get_config

os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
common_config = get_config("test_yaml")


def objective(
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


def main_tuning_func():
    search_space = {
        param: getattr(tune, spec[0])(*spec[1:])
        for param, spec in common_config.search_space.items()
    }
    reporter = CLIReporter(
        parameter_columns=list(search_space.keys())[:4] + [TRIAL_INDEX],
        metric_columns=list(set(common_config.reporter.metric_columns)),
        sort_by_metric=True,
        metric=common_config.reporter.sort_metric,
        mode="max",
        max_report_frequency=common_config.reporter.report_frequency,
    )
    search_algo = HyperOptSearch(
        points_to_evaluate=[common_config.initial_values],
        mode="max",
        metric=common_config.searcher.metric,
    )

    repeater = Repeater(
        search_algo, set_index=True, repeat=common_config.searcher.repeats
    )
    callback = GroupRewardCallback(repeater=repeater, mode="mean/var")

    # trainable_with_resources = tune.with_resources(objective, {"cpu": 2})
    tuner = tune.Tuner(
        trainable=objective,
        param_space=search_space,
        run_config=RunConfig(
            name=common_config.tuner.name,
            local_dir=os.getenv("COMMON_PATH"),
            progress_reporter=reporter,
            callbacks=[callback],
            # verbose=0,
        ),
        tune_config=tune.TuneConfig(
            num_samples=common_config.tuner.samples, search_alg=repeater
        ),
    )
    tuner.fit()


def test_search_space():
    def get_other_objs(current_key, search_space):
        other_objs = {}
        for k, v in search_space.items():
            if k != current_key:
                val = v[1]
                if hasattr(val, "__iter__"):
                    other_objs[k] = val[0]
                else:
                    other_objs[k] = val
        return other_objs

    search_space = common_config.search_space
    # try to load the env using all items of search space

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

    for k, v in search_space.items():
        val_dict = {}
        val = v[1]
        # print(k)
        other_objs = get_other_objs(k, search_space)
        if hasattr(val, "__iter__"):
            for i in val:
                val_dict[k] = i
                val_dict.update(other_objs.copy())
                # print(val_dict)
        else:
            val_dict[k] = val
            val_dict.update(other_objs.copy())

        try:
            model = load_trained_model(
                load_model_name,
                base_venv,
                model_kwargs=val_dict,
            )
            print(f"success on {k}")
        except Exception as e:
            print(f"failed on {k}")
            # print(e)
            pass


if __name__ == "__main__":
    # test_search_space()
    main_tuning_func()
