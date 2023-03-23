from imitation.algorithms.dagger import BetaSchedule
import numpy as np
from imitation.algorithms import bc
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
    VecMonitor,
)
from stable_baselines3.common.evaluation import evaluate_policy


class ExponentialBetaSchedule(BetaSchedule):
    """Exponentially-decreasing schedule for beta."""

    def __init__(self, decay: float = 0.5):
        """Builds ExponentialBetaSchedule.
        Args:
            rampdown_rounds: number of rounds over which to anneal beta.
        """
        self.decay = decay

    def __call__(self, round_num: int) -> float:
        """Computes beta value.
        Args:
            round_num: the current round number.
        Returns:
            beta linearly decreasing from `1` to `0` between round `0` and
            `self.rampdown_rounds`. After that, it is 0.
        """
        assert round_num >= 0
        return min(1, max(0, pow(self.decay, round_num)))


class BCEvalCallback:
    """
    Callback to early-stop BC training when the evaluation metric does not improve after
    a given number of consecutive checks (defined by `patience`).
    """

    def __init__(
        self,
        trainer: bc.BC,
        env,
        n_episodes,
        freq=20,
        wait: int = 3,
        patience: int = 4,
        verbose=False,
    ):
        self.trainer = trainer
        self.patience = patience
        self.best_reward = -np.inf
        self.freq = freq
        self.wait = wait
        self.env = env
        self.n_episodes = n_episodes
        self.count = 0
        self.verbose = verbose

    def __call__(self):
        if self.count == 0 or self.count % self.freq != 0:
            self.count += 1
            return False  # don't stop training
        self.count += 1
        # don't start evals until "wait" freq periods have passed
        if self.count < self.wait * self.freq:
            return False
        # if isinstance(self.trainer.venv, VecNormalize):
        #     try:
        #         sync_envs_normalization(self.trainer.venv, self.env)
        #     except AttributeError as e:
        #         raise AssertionError(
        #             "Training and eval env are not wrapped the same way, "
        #             "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
        #             "and warning above."
        #         ) from e
        reward, _ = evaluate_policy(
            self.trainer.policy,  # type: ignore[arg-type]
            self.env,
            n_eval_episodes=self.n_episodes,
            render=False,
        )
        # print(reward)
        if self.best_reward < reward:  # improved
            self.best_reward = reward
            self.wait = self.patience
            if self.verbose:
                print(f"Best reward updated: {reward:.2f}")
            return False
        else:  # no improvement
            self.wait -= 1
            if self.wait <= 0:
                # if self.verbose:
                # print(
                #     f"Stopping training because reward did not improve for {self.patience} consecutive checks"
                # )
                return True
            else:
                return False


# import json


# def check_config(config):
#     path = "/home/juuso/Documents/gradu/params/test.json"

#     # read list of dicts from json file, if config exists raise an error, else add config to json file
#     with open(path, "r") as f:
#         configs = json.load(f)
#         if config in configs:
#             raise ValueError(f"Config {config} already exists")
#         else:
#             configs.append(config)
#             configs = json.dumps(configs)


# check_config({"mode": "full", "isActive": False})
