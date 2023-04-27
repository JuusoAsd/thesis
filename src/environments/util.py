import logging
from stable_baselines3.common.vec_env import VecNormalize
from environments.mm_env_vec import MMVecEnv, SBMMVecEnv
from environments.env_configs.rewards import *
from environments.env_configs.spaces import (
    ActionSpace,
    LinearObservationSpaces,
    LinearObservation,
)
from ray.tune.search.repeater import Repeater
from ray.tune.search import Searcher

logger = logging.getLogger(__name__)
from typing import Dict, List, Optional


def setup_venv(
    data,
    obs_space=LinearObservation(LinearObservationSpaces.SimpleLinearSpace),
    act_space=ActionSpace.NormalizedAction,
    reward_class=InventoryIntegralPenalty,
    normalize=False,
    time_envs=1,
    inv_envs=1,
    env_params={},
    venv_params={},
):
    column_mapping = {col: n for (n, col) in enumerate(data.columns)}
    env = MMVecEnv(
        data.to_numpy(),
        params={
            "observation_space": obs_space,
            "action_space": act_space,
        },
        column_mapping=column_mapping,
        reward_class=reward_class,
        time_envs=time_envs,
        inv_envs=inv_envs,
        **env_params,
    )
    venv = SBMMVecEnv(env, **venv_params)

    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=100_000)
    return venv


# class TrialRepeater(Repeater):
#     def __init__(self, searcher: Searcher, repeat: int = 1, set_index: bool = True):
#         super().__init__(searcher, repeat, set_index)

#     def on_trial_complete(self, trial_id: str, result: Optional[Dict] = None, **kwargs):
#         """Stores the score for and keeps track of a completed trial.

#         Stores the metric of a trial as nan if any of the following conditions
#         are met:

#         1. ``result`` is empty or not provided.
#         2. ``result`` is provided but no metric was provided.

#         """
#         if trial_id not in self._trial_id_to_group:
#             logger.error(
#                 "Trial {} not in group; cannot report score. "
#                 "Seen trials: {}".format(trial_id, list(self._trial_id_to_group))
#             )
#         trial_group = self._trial_id_to_group[trial_id]
#         if not result or self.searcher.metric not in result:
#             score = np.nan
#         else:
#             score = result[self.searcher.metric]
#         trial_group.report(trial_id, score)
#         if trial_group.finished_reporting():
#             scores = trial_group.scores()
#             print(f"Trial {trial_id} finished reporting, scores: {np.nanmean(scores)}")
#             self.searcher.on_trial_complete(
#                 trial_group.primary_trial_id,
#                 result={self.searcher.metric: np.nanmean(scores)},
#                 **kwargs,
#             )

#             # for trial_id in trial_group._trials:
#             #     print(trial_id.values())
