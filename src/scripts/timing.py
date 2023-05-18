import logging
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from src.data_management import get_data_by_dates

from dotenv import load_dotenv
import argparse

load_dotenv(".env")

from src.util import get_test_config
from src.environments.util import setup_venv_config
from src.environments.env_configs.policies import ASPolicyVec

import time


def profile_expert():
    config = get_test_config("profiling")
    start_time = time.time()
    venv = setup_venv_config(config.data, config.env, config.venv)
    obs = venv.reset()
    expert_policy = ASPolicyVec(venv.env, **config.expert_params)
    action_func = expert_policy.get_action_func()
    n = 0
    max_steps = 1000000
    while True:
        n += 1
        if n % 10_000 == 0:
            print(n)
        action = action_func(obs)
        obs, _, done, _ = venv.step(action)
        if np.any(done) or n > max_steps:
            break

    print(f"Time: {time.time() - start_time}")


def profile_learning():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose profiling method.")
    parser.add_argument("method", type=str, help='either "expert" or "learning"')

    args = parser.parse_args()

    func_dict = {
        "profile_expert": profile_expert,
        "profile_learning": profile_learning,
    }

    try:
        func = func_dict[args.method]
    except KeyError as e:
        raise (
            f"Unknown method: {args.method}. Please choose either from {list(func_dict.keys())}"
        )
