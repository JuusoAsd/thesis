import os
import sys

sys.path.append(f"{os.getcwd()}/gradu")

import shutil
from omegaconf import OmegaConf
from ray import tune
from ray.tune import CLIReporter
from ray.air.config import RunConfig

# from ray.tune.search.repeater import Repeater
from src.environments.env_configs.callbacks import CustomRepeater
from ray.tune.search.hyperopt import HyperOptSearch

from src.util import (
    create_parameter_space,
    flatten_config,
    get_config,
    trial_namer,
    check_config_null,
)
from src.tuning_func import objective_preload_repeat
from src.environments.env_configs.callbacks import GroupRewardCallback

os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"


def tune_action_func_selection(config):
    if config.trial_run:
        short_override = get_config("short_run_override")
        config = OmegaConf.merge(config, short_override)
    check_config_null(config)
    search_space = create_parameter_space(config.search_space)
    flat_config = flatten_config(search_space)

    clear_previous = True
    if clear_previous:
        path = os.path.join(os.getenv("TRIALS"), config.run_name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        print("removed previous trials")
    reporter = CLIReporter(
        parameter_columns=config.reporter.parameter_columns,
        metric_columns=config.reporter.metric_columns,
        sort_by_metric=True,
        metric=config.reporter.sort_metric,
        mode="max",
        max_report_frequency=config.reporter.report_frequency,
        max_progress_rows=config.reporter.report_rows,
    )

    search_algo = HyperOptSearch(
        mode="max",
        metric=config.searcher.metric,
    )
    repeater = CustomRepeater(
        search_algo,
        set_index=True,
        repeat=config.searcher.repeats,
        aggregation=config.searcher.aggregation,
    )
    callback = GroupRewardCallback(repeater=repeater, mode=config.searcher.aggregation)

    trainable = objective_preload_repeat
    tuner = tune.Tuner(
        trainable=trainable,
        param_space=flat_config,
        run_config=RunConfig(
            name=config.run_name,
            local_dir=os.getenv("TRIALS"),
            progress_reporter=reporter,
            verbose=1,
            callbacks=[callback],
        ),
        tune_config=tune.TuneConfig(
            num_samples=config.samples,
            search_alg=repeater,
            trial_name_creator=trial_namer,
        ),
    )

    tuner.fit()


def tune_override(base_config, override_config):
    base = get_config(base_config)
    override = get_config(override_config)
    config = OmegaConf.merge(base, override)
    print(f"Running: {config.run_name}")
    tune_action_func_selection(config)


def tune_config(config):
    config = get_config(config)
    tune_action_func_selection(config)


if __name__ == "__main__":
    tune_override("tuning_preload_single_reward_base", "pnl_override")
