import logging
from typing import Dict, Optional
from ray import tune
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.environments.env_configs.policies import ASPolicyVec
from src.cloning import save_trained_model
from ray.tune.search.repeater import Repeater
from ray.tune.search import Searcher

logger = logging.getLogger(__name__)


class ExternalMeasureCallback(BaseCallback):
    """
    Custom callback used in the training loop to measure the performance of the agent using financial metrics
    Used to determine if training should be stopped early when agent is not improving
    Can handle validation in multiple paths, (time, inventory, or both)
        - the reward is then measured as average of the metrics over all paths
    """

    def __init__(
        self,
        data,
        venv,
        model_name="",
        patience=6,
        wait=15,
        freq=10,
        improvement_thresh=0.1,
        initial_expert=False,
        save_best_model=False,
        time_envs=1,
        inv_envs=1,
        time_data_portion=0.5,
        inv_jump=0.18,
        verbose: int = 0,
        eval_mode="min_sharpe",
    ):
        super(ExternalMeasureCallback, self).__init__(verbose)
        self.data = data
        # create a copy of the venv
        self.venv = venv.clone_venv(
            self.data,
            time_envs=time_envs,
            inv_envs=inv_envs,
            inv_jump=inv_jump,
            data_portion=time_data_portion,
        )
        self.model_name = model_name
        self.save_model = save_best_model
        if self.save_model:
            assert self.model_name != "", "Model name must be provided when saving"
        self.init_patience = patience
        self.patience = patience
        self.wait = wait
        self.freq = freq
        self.improvement_thresh = improvement_thresh
        self.eval_count = 0
        self.continue_training = True
        self.eval_mode = eval_mode
        if initial_expert:
            self.get_expert_performance()
        else:
            self.best_reward = None
        self.best_performance_metrics = {}

    def get_expert_performance(self):
        expert_policy = ASPolicyVec
        expert_params = {
            "max_order_size": 5,
            "tick_size": 0.0001,
            "max_ticks": 10,
            "price_decimals": 4,
            "inventory_target": 0,
            "risk_aversion": 0.2,
            "order_size": 1,
            "obs_type": self.venv.env.obs_space,
            "act_type": self.venv.env.act_space,
        }
        self.expert_policy = expert_policy(env=self.venv.env, **expert_params)
        self.expert_func = self.expert_policy.get_action_func()
        obs = self.venv.reset()
        while True:
            act = self.expert_func(obs)
            obs, _, done, _ = self.venv.step(act)
            if done:
                break
        self.best_reward = self.get_performance(self.venv.env)
        self.venv.reset()

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if not self.continue_training:
            logging.info("stopping training after no improvements")
        return self.continue_training

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # For each rollout, we want to measure the FINANCIAL performance of the agent
        # If the agent is not improving, we want to stop training early
        self.eval_count += 1
        # only evaluate after "wait" freq periods have passed
        if self.eval_count > self.wait:
            if self.eval_count % self.freq == 0:
                # measure agent performance
                self.venv.env.reset_metrics()
                obs = self.venv.reset()

                while True:
                    act = self.locals["self"].predict(obs, deterministic=True)[0]
                    obs, _, done, _ = self.venv.step(act)
                    if np.any(done):
                        break

                performance_metrics = self.venv.env.get_metrics()
                if self.best_performance_metrics == {}:
                    self.best_performance_metrics = {
                        key: 0 for key in performance_metrics.keys()
                    }
                agent_reward = self.get_performance(performance_metrics)
                if self.best_reward is None:
                    self.best_reward = agent_reward
                    self.best_performance_metrics = {
                        k: np.mean(v) for k, v in performance_metrics.items()
                    }
                elif agent_reward > self.best_reward * (1 + self.improvement_thresh):
                    self.patience = self.init_patience
                    self.best_reward = agent_reward
                    self.best_performance_metrics = {
                        k: np.mean(v) for k, v in performance_metrics.items()
                    }
                    if self.save_model:
                        save_trained_model(self.locals["self"], self.model_name)
                else:
                    self.patience -= 1
                    if self.patience <= 0:
                        self.continue_training = False
                        print(f"stopping training after {self.eval_count} evals")
                mean_performance_metrics = {
                    k: np.mean(v) for k, v in performance_metrics.items()
                }
                logging.debug(
                    f"Evals: {self.eval_count}, patience: {self.patience} agent reward: {agent_reward}, best reward: {self.best_reward}, {mean_performance_metrics}"
                )

        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        return True

    def get_performance(self, metrics):
        if self.eval_mode == "min_sharpe":
            is_liquidated = metrics["max_inventory"] > 0.99
            aggregate = np.minimum(
                metrics["sharpe"], metrics["sharpe"] * (1 - is_liquidated)
            )
            return np.min(aggregate)
        elif self.eval_mode == "mean_sharpe":
            is_liquidated = metrics["max_inventory"] > 0.99
            aggregate = np.mean(
                [metrics["sharpe"], metrics["sharpe"] * (1 - is_liquidated)]
            )
            return aggregate
        elif self.eval_mode == "return-inventory**2":
            is_liquidated = metrics["max_inventory"] > 0.99
            metric = metrics["episode_return"] - metrics["mean_abs_inv"] ** 2
            aggregate = np.minimum(metric, metric * (1 - is_liquidated))
            return np.min(aggregate)
        elif self.eval_mode == "return/inventory":
            is_liquidated = metrics["max_inventory"] > 0.99
            metric = metrics["episode_return"] / (metrics["mean_abs_inv"] + 1e-6)
            aggregate = np.minimum(metric, metric * (1 - is_liquidated))
            return np.min(aggregate)
        else:
            raise NotImplementedError


class GroupRewardCallback(tune.Callback):
    """
    callback keeps track of when trial group finishes and the reports the average result
    """

    def __init__(self, repeater, mode="mean"):
        self.results = {}
        self.repeater = repeater
        self.groups = {}
        self.mode = mode
        self.finished = 0

    def on_trial_complete(self, iteration, trials, trial):
        group_id = self.repeater._trial_id_to_group[trial.trial_id]

        if group_id not in self.groups:
            self.groups[group_id] = {
                "rewards": [trial.last_result["trial_reward"]],
                "count": 1,
                "trials": [trial],
            }
        else:
            self.groups[group_id]["rewards"].append(trial.last_result["trial_reward"])
            self.groups[group_id]["count"] += 1
            self.groups[group_id]["trials"].append(trial)

        if self.groups[group_id]["count"] == self.repeater.repeat:
            if self.mode == "mean":
                result = np.mean(self.groups[group_id]["rewards"])
            elif self.mode == "mean/var":
                result = np.mean(self.groups[group_id]["rewards"]) / (
                    1 + np.var(self.groups[group_id]["rewards"])
                )
            elif self.mode == "min":
                result = np.min(self.groups[group_id]["rewards"])
            for trial in self.groups[group_id]["trials"]:
                trial.last_result["group_reward"] = result


class CustomRepeater(Repeater):
    def __init__(
        self,
        searcher: Searcher,
        repeat: int = 1,
        set_index: bool = True,
        aggregation="mean",
    ):
        super(CustomRepeater, self).__init__(searcher, repeat)
        self.aggregation = aggregation

    def on_trial_complete(self, trial_id: str, result: Optional[Dict] = None, **kwargs):
        if trial_id not in self._trial_id_to_group:
            logger.error(
                "Trial {} not in group; cannot report score. "
                "Seen trials: {}".format(trial_id, list(self._trial_id_to_group))
            )
        trial_group = self._trial_id_to_group[trial_id]
        if not result or self.searcher.metric not in result:
            score = np.nan
        else:
            score = result[self.searcher.metric]
        trial_group.report(trial_id, score)

        if trial_group.finished_reporting():
            scores = trial_group.scores()
            if self.aggregation == "mean":
                metric = np.nanmean(scores)
            elif self.aggregation == "min":
                metric = np.nanmin(scores)
            elif self.aggregation == "mean/var":
                metric = np.nanmean(scores) / (1 + np.nanvar(scores))
            else:
                raise NotImplementedError
            self.searcher.on_trial_complete(
                trial_group.primary_trial_id,
                result={self.searcher.metric: metric},
                **kwargs,
            )
