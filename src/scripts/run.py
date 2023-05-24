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


if __name__ == "__main__":
    func_dict = {
        "profile_expert": profile_expert,
        "profile_learning": profile_learning,
        "tune_config": tune_config,
        "tune_override": tune_override,
        "create_cloned_models": create_cloned_models,
    }

    parser = argparse.ArgumentParser(description="Choose profiling method.")
    parser.add_argument("method", type=str, help=f"Select from {func_dict.keys()}")
    parser.add_argument(
        "args", nargs=argparse.REMAINDER, help="Positional arguments for the function"
    )
    args = parser.parse_args()
    try:
        func = func_dict[args.method]
        func(*args.args)
    except KeyError as e:
        raise ValueError(
            f"Unknown method: {args.method}. Please choose either from {list(func_dict.keys())}"
        )
    except Exception as e:
        raise e
