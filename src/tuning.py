import os
import shutil
from omegaconf import OmegaConf
from ray import tune
from ray.tune import CLIReporter
from ray.air.config import RunConfig
from ray.tune.search.repeater import Repeater, TRIAL_INDEX
from ray.tune.search.hyperopt import HyperOptSearch

from src.util import get_config
from src.tuning_func import objective_clone, objective_preload_repeat
from src.util import create_parameter_space, flatten_config
from src.environments.env_configs.callbacks import GroupRewardCallback

os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
common_config = get_config("test_yaml")


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
        max_progress_rows=common_config.reporter.max_progress_rows,
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


def cont_tuning():
    tuner = tune.Tuner.restore("/Volumes/ssd/gradu_data/wide_tuning_sharpe")
    tuner.fit()


def trial_namer(trial):
    return f"{trial.trainable_name}_{trial.trial_id}"


def tune_cloning():
    config = get_config("tuning_train_eval_multi_run_rewards")
    search_space = create_parameter_space(config.search_space)
    flat_config = flatten_config(search_space)

    reporter = CLIReporter(
        # parameter_columns=list(search_space.keys())[:1],
        parameter_columns=config.reporter.parameter_columns,
        metric_columns=config.reporter.metric_columns,
        sort_by_metric=True,
        metric=config.reporter.sort_metric,
        mode="max",
        max_report_frequency=config.reporter.report_frequency,
        max_progress_rows=config.reporter.report_rows,
    )

    search_algo = HyperOptSearch(
        # points_to_evaluate=[config.searcher.initial_values],
        mode="max",
        metric=config.searcher.metric,
    )

    # trainable_with_resources = tune.with_resources(objective, {"cpu": 2})
    tuner = tune.Tuner(
        trainable=objective_clone,
        param_space=flat_config,
        run_config=RunConfig(
            name=config.run_name,
            local_dir=os.getenv("TRIALS"),
            progress_reporter=reporter,
            verbose=1,
        ),
        tune_config=tune.TuneConfig(
            num_samples=config.samples,
            search_alg=search_algo,
            trial_name_creator=trial_namer,
        ),
    )

    tuner.fit()


def tune_pre_cloned_choose():
    config = get_config("tuning_preload_repeat")
    short_override = get_config("short_run_override")
    config = OmegaConf.merge(config, short_override)
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
    repeater = Repeater(search_algo, set_index=True, repeat=config.searcher.repeats)
    callback = GroupRewardCallback(repeater=repeater, mode="mean/var")

    tuner = tune.Tuner(
        trainable=objective_preload_repeat,
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


if __name__ == "__main__":
    # test_search_space()
    # main_tuning_func()
    # cont_tuning()
    # tune_cloning()
    tune_pre_cloned_choose()
