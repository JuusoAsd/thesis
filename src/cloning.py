import os
from enum import Enum
import tempfile

from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from stable_baselines3 import PPO
import torch as th
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data import rollout
from stable_baselines3.common.vec_env import VecNormalize


from environments.env_configs.spaces import ActionSpace, ObservationSpace
from environments.env_configs.policies import (
    ASPolicyVec,
    AlwaysSamePolicyVec,
    SimplePolicyVec,
    RoundingPolicyVec,
    NThDigitPolicyVec,
)
from environments.mm_env_vec import MMVecEnv, SBMMVecEnv
from environments.env_configs.util import ExponentialBetaSchedule, BCEvalCallback

from environments.env_configs.rewards import *
import environments.env_configs.policies as policies


load_dotenv()

base = pd.read_csv(os.getenv("BASE_PATH"))
indicators = pd.read_csv(os.getenv("INDICATOR_PATH"))
data = pd.merge(base, indicators, on="timestamp", how="left").ffill().dropna()
data["mid_price"] = np.round((data.best_bid + data.best_ask) / 2, 5)

column_mapping = {col: n for (n, col) in enumerate(data.columns)}
rng = np.random.default_rng(0)


def clone_simple_policy_dagger():
    n_env = 8
    obs_space = ObservationSpace.SimpleObservation
    act_space = ActionSpace.NormalizedAction

    env = MMVecEnv(
        data.to_numpy(),
        n_envs=n_env,
        params={
            "observation_space": obs_space,
            "action_space": act_space,
        },
        column_mapping=column_mapping,
        reward_class=InventoryIntegralPenalty,
    )

    venv = SBMMVecEnv(env)
    student_model = PPO("MlpPolicy", venv, verbose=1)
    student_policy = student_model.policy
    expert_trainer = SimplePolicyVec(env)
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
            n_episodes=25,  # how many episodes to evaluate on
        )

        imitation_trainer.train(
            1_000_000,
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

    save_trained_model(student_model, "clone_simple_dagger")


def save_trained_model(model, path):
    """
    Save trained model optionally with its replay buffer
    and ``VecNormalize`` statistics

    model: the trained model
    dir: optional sub-dir of self.save_path to save the model in
    """
    model_dir = Path(f"{os.getcwd()}/models")

    model_path = str(model_dir / path)
    model.save(model_path)

    wrapper = model.get_vec_normalize_env()
    if wrapper is not None:
        wrapper.save(str(model_dir / (path + "_vec_normalize.pkl")))


def load_trained_model(name, venv, normalize=True):
    model_dict = {}
    model_dir = Path(f"{os.getcwd()}/models")

    model_path = str(model_dir / name)
    normalize_path = str(model_dir / (name + "_vec_normalize.pkl"))
    if os.path.exists(normalize_path) and normalize:
        normalized_venv = VecNormalize.load(normalize_path, venv)
        model = PPO.load(model_path, env=normalized_venv)
    else:
        model = PPO.load(model_path)

    return model


class Cloning(Enum):
    BC = 1
    Dagger = 2


class CloneDuration(Enum):
    Short = 1
    Long = 2
    VeryLong = 3


def clone_bc(venv, expert_trainer, student_model, duration, random=True):
    env = venv.env

    # depending on what duration is, there are 2 different configs
    config_dict = {
        CloneDuration.Short: {
            "min_episodes": 10_000,
            "evaluation_freq": 1000,
            "n_batches": 5000,
        },
        CloneDuration.Long: {
            "min_episodes": 1_000_000,
            "evaluation_freq": 25_000,
            "n_batches": 50_000,
        },
        CloneDuration.VeryLong: {
            "min_episodes": 10_000_000,
            "evaluation_freq": 25_000,
            "n_batches": 500_000,
        },
    }
    chosen_config = config_dict[duration]

    rollouts = rollout.rollout(
        expert_trainer.get_action_func(),
        venv,
        rollout.make_sample_until(
            min_timesteps=None, min_episodes=chosen_config["min_episodes"]
        ),
        rng=rng,
        unwrap=False,
    )

    transitions = rollout.flatten_trajectories(rollouts)

    # TESTING
    # obs_0 = transitions.obs[0]
    # # unnorm = venv.unnormalize_obs(obs_0)

    # print(np.min(transitions.obs, axis=0))
    # print(np.max(transitions.obs, axis=0), "\n")

    # print(np.min(transitions.acts, axis=0))
    # print(np.max(transitions.acts, axis=0), "\n")
    # n = 0
    # all_val = []
    # while True:
    #     try:
    #         obs = transitions.obs[n]
    #         # obs_unnorm = venv.unnormalize_obs(obs)
    #         act = transitions.acts[n]

    #         print(f"obs:    {np.round(transitions.obs[n],4).tolist()}")
    #         print(f"acts:   {np.round(transitions.acts[n],4).tolist()}\n")

    #         # current_val = obs.tolist() + obs_unnorm.tolist() + act.tolist()
    #         current_val = obs.tolist() + act.tolist()
    #         all_val.append(current_val)

    #         n += 1
    #     except IndexError:
    #         break
    # print(
    #     pd.DataFrame(all_val).to_csv(
    #         "/Users/juusoahlroos/Documents/own/gradu/src/data_management/clone_data.csv",
    #         index=False,
    #     )
    # )
    # # start = True
    # # for i in transitions.obs:

    # #     print(i)
    # exit()

    # print(f"norm obs: {obs_0}")
    # print(f"unnorm obs: {unnorm}")

    # END TESTING
    student_policy = student_model.policy
    imitation_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=student_policy,  # USE SAME ARCHITECTURE AS MODEL!!!
        demonstrations=transitions,
        batch_size=32,
        optimizer_cls=th.optim.Adam,
        optimizer_kwargs=dict(lr=0.001),
        ent_weight=1e-3,
        l2_weight=0.0,
        rng=rng,
    )
    eval_callback = BCEvalCallback(
        imitation_trainer,
        venv,
        freq=chosen_config["evaluation_freq"],  # how often to evaluate
        wait=20,  # how long to wait until first evaluation
        patience=10,
        n_episodes=20,  # how many episodes to evaluate on
    )
    imitation_trainer.train(
        n_batches=chosen_config["n_batches"],
        log_interval=10_000,
        log_rollouts_venv=venv,
        log_rollouts_n_episodes=1,
        on_batch_end=eval_callback,
    )

    save_trained_model(student_model, "clone_bc")
    return "clone_bc"


def compare_cloned(env, model, expert_policy, action_count=100, normalize=True):
    env.n_env = 1
    if type(model) == str:
        model = load_trained_model(model, env, normalize=normalize)
    # first run 10 random observations and compare them to expert
    expert = expert_policy(model.env)
    expert_func = expert.get_action_func()
    if normalize:
        for i in range(10):
            obs = env.observation_space.sample().reshape(1, -1)
            norm_obs = model.env.normalize_obs(obs)
            action_model, _states = model.predict(obs, deterministic=True)
            expert_action = expert_func(norm_obs)
            print(f"Observation:    {obs}")
            print(f"Normalized:     {norm_obs}")
            print(f"Action:         {action_model}")
            print(f"Expert action:  {expert_action}\n")

        # then run 10 predictions against the expert and see if answer is always the same
        obs = env.observation_space.sample().reshape(1, -1)
        prev_act = None
        act_count = 0
        for i in range(10):
            action, _ = model.predict(obs, deterministic=True)
            if action.tolist() != prev_act:
                act_count += 1
        if act_count > 1:
            print(f"more than one action, actions: {act_count}")

        # then run 10 steps with both model and expert and see the result
        n = 0
        wrong_actions = 0
        disctint_actions = 0
        previous_action = None
        obs = env.reset()
        action_count = 10
        while True:
            # this should already be normalized
            print(f"step obs: {obs}")
            n += 1
            action_model, _states = model.predict(obs, deterministic=True)
            action_expert = expert_func(obs)

            print(f"action model: {action_model}")
            print(f"action expert: {action_expert} \n")
            if n >= action_count:
                break
            obs, _, _, _ = env.step(action_model)
        print(f"distinct action share: {disctint_actions/n*100}%")
    else:
        for i in range(10):
            obs = env.observation_space.sample().reshape(1, -1)
            action_model, _states = model.predict(obs, deterministic=False)
            expert_action = expert_func(obs)
            print(f"Observation:    {obs}")
            print(f"Action:         {action_model}")
            print(f"Expert action:  {expert_action}\n")

        # then run 10 predictions against the expert and see if answer is always the same
        obs = env.observation_space.sample().reshape(1, -1)
        prev_act = None
        act_count = 0
        for i in range(10):
            action, _ = model.predict(obs, deterministic=True)
            if action.tolist() != prev_act:
                act_count += 1
        if act_count > 1:
            print(f"more than one action, actions: {act_count}")

        # then run 10 steps with both model and expert and see the result
        n = 0
        wrong_actions = 0
        disctint_actions = 0
        previous_action = None
        obs = env.reset()
        action_count = 10
        while True:
            # this should already be normalized
            print(f"step obs: {obs}")
            n += 1
            action_model, _states = model.predict(obs, deterministic=True)
            action_expert = expert_func(obs)

            print(f"action model: {action_model}")
            print(f"action expert: {action_expert} \n")
            if n >= action_count:
                break
            obs, _, _, _ = env.step(action_model)
        print(f"distinct action share: {disctint_actions/n*100}%")


def model_cloning():
    # setup model for cloning here
    # parameters are set at the beginning of the function
    clone = True
    model_name = "clone_bc"
    n_env = 5
    normalize = False
    obs_space = ObservationSpace.SimpleObservation
    act_space = ActionSpace.NormalizedAction
    cloning_model = Cloning.BC
    cloning_duration = CloneDuration.VeryLong
    expert_policy = NThDigitPolicyVec

    env = MMVecEnv(
        data.to_numpy(),
        n_envs=n_env,
        params={
            "observation_space": obs_space,
            "action_space": act_space,
        },
        column_mapping=column_mapping,
        reward_class=InventoryIntegralPenalty,  # reward should not matter?
    )

    venv = SBMMVecEnv(env)
    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=100_000)
    student_model = PPO("MlpPolicy", venv, verbose=1)
    expert = expert_policy(venv)

    if clone:
        if cloning_model == Cloning.BC:
            model_name = clone_bc(venv, expert, student_model, cloning_duration)
    compare_cloned(venv, model_name, expert_policy, action_count=3, normalize=normalize)


def test_normalized_env():
    model_name = "clone_bc"
    n_env = 1
    obs_space = ObservationSpace.SimpleObservation
    act_space = ActionSpace.NormalizedAction
    env = MMVecEnv(
        data.to_numpy(),
        n_envs=n_env,
        params={
            "observation_space": obs_space,
            "action_space": act_space,
        },
        column_mapping=column_mapping,
        reward_class=InventoryIntegralPenalty,  # reward should not matter?
    )

    cloning_model = Cloning.BC
    cloning_duration = CloneDuration.Long
    venv = SBMMVecEnv(env)
    venv_wrapped = VecNormalize(
        venv, norm_obs=True, norm_reward=False, clip_obs=100_000
    )
    student_model = PPO("MlpPolicy", venv_wrapped, verbose=1)
    expert = SimplePolicyVec(venv_wrapped)
    action_func = expert.get_action_func()

    intensity = 50_000
    vol = 0.5
    # should return:
    # abs(intensity / 100k)
    # abs(intensity / 200k)
    # vol / 2
    # vol ** 2

    # however, as normalized values are clipped between min and max [-10, 10], it is not so
    # if change clipping to 100k, works as expected

    obs_unnorm = np.array([0, vol, intensity])
    obs_norm = venv_wrapped.normalize_obs(obs_unnorm)
    action_unnorm = action_func(obs_norm)
    act = action_unnorm
    print(
        f"expected: {abs(intensity / 100_000)}, {abs(intensity / 200_000)}, {vol / 2}, {vol ** 2}"
    )
    print(f"actual: {act}")


if __name__ == "__main__":
    # clone_manual_policy_dagger()
    # clone_manual_policy_bc()
    # clone_always_same_policy_bc()
    # clone_simple_policy()
    # clone_simple_policy_dagger()
    # compare_cloned()
    model_cloning()
    # test_normalized_env()
