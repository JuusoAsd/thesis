import os
import random
import time
import collections.abc as collections

from omegaconf import OmegaConf
from ray import tune
from ray.air.config import RunConfig

from src.util import get_config, create_parameter_space


def algo_params_sampler(config):
    algo = config["model"]["algo"]
    params = config["model"]["model_params"][algo]
    return params


# def create_parameter_space(config):
#     param_space = {}
#     for key, value in config.items():
#         if isinstance(value, collections.Sequence):
#             param_space[key] = getattr(tune, value[0])(*value[1:])
#         elif isinstance(value, collections.Mapping):
#             nested_space = create_parameter_space(value)
#             param_space[key] = nested_space
#         else:
#             param_space[key] = value
#     return param_space


def clone_objective(config_dict):
    config = OmegaConf.create(config_dict)

    print("Running objective function")
    print(config)
    # print(config["model"])
    print(config.model)
    time.sleep(20)
    return {"res": random.randint(0, 100)}


def main_func():
    # test_config = get_config("tuning_train_eval_single_run")
    test_config = get_config("tuning_train_eval_multi_run")
    search_space = create_parameter_space(test_config.search_space)
    tuner = tune.Tuner(
        trainable=clone_objective,
        param_space=search_space,
        run_config=RunConfig(
            name="test_trial",
            local_dir=os.getenv("COMMON_PATH"),
        ),
        tune_config=tune.TuneConfig(num_samples=20),
    )
    tuner.fit()


if __name__ == "__main__":
    main_func()
