import os
import random
import numpy as np
import torch as th  # noqa: F401
from stable_baselines3.common.utils import set_random_seed
from hydra import compose, initialize_config_dir
from dotenv import load_dotenv
from ray import tune

load_dotenv()


def set_seeds(seed):
    np.random.seed(seed)
    set_random_seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = "42"


def get_config(name):
    config_path = os.environ.get("CONFIG_PATH")
    with initialize_config_dir(config_dir=config_path):
        cfg = compose(config_name=f"{name}.yaml")
    return cfg


if __name__ == "__main__":
    config = get_config("test_yaml")
    print(config)
    print(config.description)
    print(config.reporter.metric_columns)

    search_space_spec = config.search_space
    search_space = {
        param: getattr(tune, spec[0])(*spec[1:])
        for param, spec in search_space_spec.items()
    }
    print(search_space)
