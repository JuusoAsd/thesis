import numpy as np
from imitation.algorithms.dagger import BetaSchedule
from imitation.algorithms import bc
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
        reward, _ = evaluate_policy(
            self.trainer.policy,  # type: ignore[arg-type]
            self.env,
            n_eval_episodes=self.n_episodes,
            render=False,
        )
        if self.best_reward < reward:  # improved
            self.best_reward = reward
            self.wait = self.patience
            if self.verbose:
                print(f"Best reward updated: {reward:.2f}")
            return False
        else:  # no improvement
            self.wait -= 1
            if self.wait <= 0:
                return True
            else:
                return False
