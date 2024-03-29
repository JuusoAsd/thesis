import os
import sys
import warnings


sys.path.append(f"{os.getcwd()}/gradu")

from dotenv import load_dotenv

load_dotenv(".env")

import argparse
from src.scripts.timing import profile_expert, profile_learning
from src.scripts.cloning import create_cloned_models
from src.scripts.tuning import tune_config, tune_override
from src.scripts.testing import rolling_test_train, rolling_train_test_as
from src.scripts.research import create_model_decision_grid


if __name__ == "__main__":
    func_dict = {
        "profile_expert": profile_expert,
        "profile_learning": profile_learning,
        "tune_config": tune_config,
        "tune_override": tune_override,
        "create_cloned_models": create_cloned_models,
        "rolling_test_train": rolling_test_train,
        "decision_grid": create_model_decision_grid,
        "test_as": rolling_train_test_as,
    }

    parser = argparse.ArgumentParser(description="Choose profiling method.")
    parser.add_argument("method", type=str, help=f"Select from {func_dict.keys()}")
    parser.add_argument(
        "args", nargs=argparse.REMAINDER, help="Positional arguments for the function"
    )
    args = parser.parse_args()
    try:
        func = func_dict[args.method]
    except KeyError as e:
        raise ValueError(
            f"Unknown method: {args.method}. Please choose either from {list(func_dict.keys())}"
        )
    except Exception as e:
        raise e

    try:
        func(*args.args)
    except Exception as e:
        warnings.warn(f"Error in {args.method} while running: {e}")
        raise e
