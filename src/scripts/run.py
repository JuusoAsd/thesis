import os
import sys


sys.path.append(f"{os.getcwd()}/gradu")

from dotenv import load_dotenv
load_dotenv(".env")

import argparse
from src.scripts.timing import  profile_expert, profile_learning
from src.scripts.cloning import create_cloned_models
from src.scripts.tuning import tune_action_func_selection



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Choose profiling method.")
    parser.add_argument("method", type=str, help='either "expert" or "learning"')

    args = parser.parse_args()

    func_dict = {
        "profile_expert": profile_expert,
        "profile_learning": profile_learning,
        "tune_action_func_selection": tune_action_func_selection,
        "create_cloned_models": create_cloned_models,
    }

    try:
        func = func_dict[args.method]()
    except KeyError as e:
        raise (
            f"Unknown method: {args.method}. Please choose either from {list(func_dict.keys())}"
        )