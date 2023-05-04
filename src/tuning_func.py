import os
from datetime import datetime, timedelta
import logging

from ray.tune.search.repeater import TRIAL_INDEX
from stable_baselines3 import PPO


# from testing import test_trained_vs_manual
from cloning import (
    clone_expert_config,
    get_transitions,
    load_trained_model,
    load_model_by_config,
)
from environments.util import setup_venv, setup_venv_config
from data_management import get_data_by_dates
from util import set_seeds, get_config, filter_config_for_class
from environments.env_configs.spaces import ActionSpace
from environments.env_configs.rewards import AssymetricPnLDampening
from environments.env_configs.callbacks import ExternalMeasureCallback


os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
common_config = get_config("test_yaml")


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


def objective_clone(config):
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
    seed = 0
    set_seeds(seed)

    # for given config, setup the environment
    venv = setup_venv_config(config.clone_data, config.env, config.venv)

    # setup the model

    algo_dict = {"PPO": PPO}
    filtered_config = filter_config_for_class(
        config.model, algo_dict[config.model.algo]
    )
    student_model = algo_dict[config.model.algo](env=venv, **filtered_config)

    # get transitions, execute cloning
    transitions = get_transitions(config, venv)
    clone_expert_config(config, venv, student_model, transitions)

    callback_data = get_data_by_dates(**config.eval_data)
    # get results from training the model
    rewards = []
    for i in range(config.tuning.trials):
        tune_venv = setup_venv_config(config.clone_data, config.env, config.venv)
        logging.info(f"Running trial {i}")
        set_seeds(i)
        model = load_model_by_config(config, tune_venv)
        callback = ExternalMeasureCallback(
            data=callback_data.to_numpy(), venv=tune_venv, **config.tuning.callback
        )
        model.learn(
            total_timesteps=config.tuning.timesteps,
            log_interval=None,
            callback=callback,
        )
        rewards.append(callback.best_reward)

    print(rewards)


class TestClass:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def get(self):
        print(self.a, self.b)


from enum import Enum


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


# cls = hydra.utils.instantiate(test_config.test_class)
# cls.get()
# chosen_color = Color[test_config.color]
# print(chosen_color)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_config = get_config("tuning_train_eval_single_run")
    objective_clone(test_config)
