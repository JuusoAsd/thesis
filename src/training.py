"""
All things related to training the model
"""
from pathlib import Path
import numpy as np
import pandas as pd

import tempfile
from stable_baselines3 import PPO
import torch as th
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer

from environments.env_configs.spaces import ActionSpace, ObservationSpace
from environments.env_configs.policies import ASPolicyVec
from environments.mm_env_vec import MMVecEnv, SBMMVecEnv
from environments.env_configs.util import ExponentialBetaSchedule, BCEvalCallback

from environments.env_configs.rewards import *

base = pd.read_csv("/home/juuso/Documents/gradu/parsed_data/aggregated/base_data.csv")
indicators = pd.read_csv(
    "/home/juuso/Documents/gradu/parsed_data/aggregated/indicator_data.csv"
)
data = pd.merge(base, indicators, on="timestamp", how="left").ffill().dropna()
data["mid_price"] = (data.best_bid + data.best_ask) / 2
column_mapping = {col: n for (n, col) in enumerate(data.columns)}

rng = np.random.default_rng(0)


def clone_manual_policy():
    n_env = 8
    env = MMVecEnv(
        data.to_numpy(),
        n_envs=n_env,
        params={
            "observation_space": ObservationSpace.OSIObservation,
            "action_space": ActionSpace.NormalizedAction,
        },
        column_mapping=column_mapping,
        reward_class=AssymetricPnLDampening,
    )
    venv = SBMMVecEnv(env)

    student_model = PPO("MlpPolicy", venv, verbose=1)
    student_policy = student_model.policy
    expert_trainer = ASPolicyVec(
        max_order_size=10,
        tick_size=0.0001,
        max_ticks=10,
        price_decimals=4,
        inventory_target=0,
        risk_aversion=0.1,
        order_size=1,
        n_env=n_env,
    )
    rng = np.random.default_rng(0)
    bc_trainer = bc.BC(
        rng=rng,
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=student_policy,
        batch_size=32,
        ent_weight=1e-3,
        l2_weight=0.0,
        optimizer_cls=th.optim.Adam,
        optimizer_kwargs=dict(lr=0.001),
    )

    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        print(tmpdir)
        imitation_trainer = SimpleDAggerTrainer(
            venv=venv,
            scratch_dir=tmpdir,
            expert_policy=expert_trainer.get_action_func(),
            bc_trainer=bc_trainer,
            rng=np.random.default_rng(15),
            beta_schedule=ExponentialBetaSchedule(0.5),
        )
        eval_callback = BCEvalCallback(
            imitation_trainer,
            venv,
            freq=5000,  # how often to evaluate
            wait=20,  # how long to wait until first evaluation
            patience=6,
            n_episodes=5,  # how many episodes to evaluate on
        )

        imitation_trainer.train(
            1000000,
            rollout_round_min_episodes=3,
            rollout_round_min_timesteps=5000,
            bc_train_kwargs=dict(
                n_batches=5000,
                on_batch_end=eval_callback,
                log_rollouts_venv=venv,  # do rollout stats on this env, not train env!
                log_rollouts_n_episodes=1,
                # progress_bar=True,
                log_interval=10_000,
            ),  # default None
        )

    save_trained_model(student_model)


def save_trained_model(model):
    """
    Save trained model optionally with its replay buffer
    and ``VecNormalize`` statistics

    model: the trained model
    dir: optional sub-dir of self.save_path to save the model in
    """
    model_dir = Path("/home/juuso/Documents/gradu/models")

    model_path = str(model_dir / "clone")
    model.save(model_path)


def load_trained_model():
    model_dir = Path("/home/juuso/Documents/gradu/models")

    model_path = str(model_dir / "clone")
    model = PPO.load(model_path)
    return model


def run_ml_cloned():
    actions = 100_000
    model = load_trained_model()
    env = MMVecEnv(
        data.to_numpy(),
        n_envs=1,
        params={
            "observation_space": ObservationSpace.OSIObservation,
            "action_space": ActionSpace.NormalizedAction,
        },
        column_mapping=column_mapping,
        reward_class=PnLReward,
    )

    obs = env.reset()
    n = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones or n > actions:
            break

    print(env.get_metrics())


def run_ml():
    train_timesteps = 100_000
    actions = 100_000
    env = SBMMVecEnv(
        MMVecEnv(
            data.to_numpy(),
            n_envs=5,
            params={
                "observation_space": ObservationSpace.OSIObservation,
                "action_space": ActionSpace.NormalizedAction,
            },
            column_mapping=column_mapping,
            reward_class=InventoryIntegralPenalty,
        )
    )

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=train_timesteps)
    obs = env.reset()
    n = 0
    print("trained")
    env.reset_envs = False
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        n += 1
        if np.all(dones) or n > actions:
            print("done", dones, n)
            break

    print(env.env.get_metrics())


def run_manual():
    actions = 100_000
    env = MMVecEnv(
        data.to_numpy(),
        n_envs=1,
        params={
            "observation_space": ObservationSpace.OSIObservation,
            "action_space": ActionSpace.NormalizedAction,
        },
        column_mapping=column_mapping,
    )
    policy = ASPolicyVec(
        max_order_size=10,
        tick_size=0.0001,
        max_ticks=10,
        price_decimals=4,
        inventory_target=0,
        risk_aversion=0.1,
        order_size=1,
    )
    obs = env.reset()
    for i in range(actions):
        action = policy.get_action(obs)
        obs, rewards, dones, info = env.step(action)

    env.get_metrics()


if __name__ == "__main__":
    # clone_manual_policy()
    # run_ml_cloned()

    # model = load_trained_model()
    # run_manual()
    run_ml()
