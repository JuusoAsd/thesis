import logging
import os
import inspect
import random
import json
import hashlib

import numpy as np
import torch as th  # noqa: F401
from stable_baselines3.common.utils import set_random_seed
from hydra import compose, initialize_config_dir
from dotenv import load_dotenv
import pandas as pd
from omegaconf import OmegaConf

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


def get_tuning_result_json(trial_name):
    path = os.getenv("COMMON_PATH")
    path = os.path.join(path, trial_name)
    # Initialize an empty list to store the results
    results = []
    config_dict = {}

    # Iterate through the subfolders
    for root, dirs, files in os.walk(path):
        for file in files:
            # Check if the file is a result.json file
            if file == "result.json":
                try:
                    # Construct the full file path
                    file_path = os.path.join(root, file)

                    # Read the JSON file
                    with open(file_path, "r") as json_file:
                        result_data = json.load(json_file)

                    # Add the result to the list
                    conf = result_data.pop("config")
                    conf.pop("__trial_index__")
                    conf_val_str = ""
                    for k, v in conf.items():
                        conf_val_str += f"{k}_{v}_"
                    if conf_val_str not in config_dict:
                        config_dict[conf_val_str] = len(config_dict)
                    result_data["trial_group_id"] = config_dict[conf_val_str]

                    for k, v in conf.items():
                        result_data[k] = v
                    results.append(result_data)
                except Exception as e:
                    print(f"Failed to load {root}")
    df = pd.DataFrame(results)
    return df


def get_tuning_result_csv(trial_name):
    path = os.getenv("COMMON_PATH")
    path = os.path.join(path, trial_name)
    # Initialize an empty dataframe to store the results
    results = pd.DataFrame()
    for root, dirs, files in os.walk(path):
        for file in files:
            # Check if the file is a result.json file
            if file == "progress.csv":
                try:
                    # Construct the full file path
                    file_path = os.path.join(root, file)

                    # Read the csv file
                    csv = pd.read_csv(file_path)
                    if results.empty:
                        results = csv
                    else:
                        results = pd.concat([results, csv])

                except Exception as e:
                    print(f"Failed to load {root}")
                    print(e)
    return results


def filter_config_for_class(config, target_class):
    # Get the parameters of the target class's __init__ method
    init_signature = inspect.signature(target_class.__init__)
    init_parameters = init_signature.parameters

    # Filter the configuration dictionary to only include the parameters accepted by the target class
    filtered_config = {
        key: value for key, value in config.items() if key in init_parameters
    }

    return filtered_config


def get_model_hash(config):
    config_str = json.dumps(
        OmegaConf.to_container(config.model, resolve=True), sort_keys=True
    )
    hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()
    logging.info(f"Current hash: {hash}")
    return hash


if __name__ == "__main__":
    # config = get_config("test_yaml")
    # print(config)
    # print(config.description)
    # print(config.reporter.metric_columns)

    # search_space_spec = config.search_space
    # search_space = {
    #     param: getattr(tune, spec[0])(*spec[1:])
    #     for param, spec in search_space_spec.items()
    # }
    # print(search_space)

    # df = get_tuning_result("tune_multiple_val_envs")
    # df.to_csv(
    #     os.path.join(os.getenv("RESULT_PATH"), "tune_multiple_val_envs.csv"),
    #     index=False,
    # )
    # print(df)

    df = get_tuning_result_csv("tune_multiple_val_envs")
    df.to_csv(
        os.path.join(os.getenv("RESULT_PATH"), "tune_multiple_val_envs_csv.csv"),
        index=False,
    )
