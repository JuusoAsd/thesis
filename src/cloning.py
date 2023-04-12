import os
from enum import Enum

from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from stable_baselines3 import PPO
import torch as th
from imitation.algorithms import bc
from imitation.data import rollout
from stable_baselines3.common.vec_env import VecNormalize


from environments.env_configs.policies import ASPolicyVec
from environments.mm_env_vec import MMVecEnv, SBMMVecEnv
from environments.env_configs.util import BCEvalCallback
from environments.env_configs.rewards import *
from environments.env_configs.spaces import (
    ActionSpace,
    LinearObservationSpaces,
    LinearObservation,
)


load_dotenv()

base = pd.read_csv(os.getenv("BASE_PATH"))
indicators = pd.read_csv(os.getenv("INDICATOR_PATH"))
data = pd.merge(base, indicators, on="timestamp", how="left").ffill().dropna()
data["mid_price"] = np.round((data.best_bid + data.best_ask) / 2, 5)

column_mapping = {col: n for (n, col) in enumerate(data.columns)}
rng = np.random.default_rng(0)


# def clone_simple_policy_dagger():
#     n_env = 8
#     obs_space = ObservationSpace.SimpleObservation
#     act_space = ActionSpace.NormalizedAction

#     env = MMVecEnv(
#         data.to_numpy(),
#         n_envs=n_env,
#         params={
#             "observation_space": obs_space,
#             "action_space": act_space,
#         },
#         column_mapping=column_mapping,
#         reward_class=InventoryIntegralPenalty,
#     )

#     venv = SBMMVecEnv(env)
#     student_model = PPO("MlpPolicy", venv, verbose=1)
#     student_policy = student_model.policy
#     expert_trainer = SimplePolicyVec(env)
#     rng = np.random.default_rng(0)
#     bc_trainer = bc.BC(
#         rng=rng,
#         observation_space=env.observation_space,
#         action_space=env.action_space,
#         policy=student_policy,
#         batch_size=32,
#         ent_weight=1e-3,
#         l2_weight=0.0,
#         optimizer_cls=th.optim.Adam,
#         optimizer_kwargs=dict(lr=0.001),
#     )

#     with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
#         print(tmpdir)
#         imitation_trainer = SimpleDAggerTrainer(
#             venv=venv,
#             scratch_dir=tmpdir,
#             expert_policy=expert_trainer.get_action_func(),
#             bc_trainer=bc_trainer,
#             rng=np.random.default_rng(15),
#             beta_schedule=ExponentialBetaSchedule(0.5),
#         )

#         eval_callback = BCEvalCallback(
#             imitation_trainer,
#             venv,
#             freq=5000,  # how often to evaluate
#             wait=20,  # how long to wait until first evaluation
#             patience=6,
#             n_episodes=25,  # how many episodes to evaluate on
#         )

#         imitation_trainer.train(
#             1_000_000,
#             rollout_round_min_episodes=3,
#             rollout_round_min_timesteps=5000,
#             bc_train_kwargs=dict(
#                 n_batches=5000,
#                 on_batch_end=eval_callback,
#                 log_rollouts_venv=venv,  # do rollout stats on this env, not train env!
#                 log_rollouts_n_episodes=1,
#                 # progress_bar=True,
#                 log_interval=10_000,
#             ),  # default None
#         )

#     save_trained_model(student_model, "clone_simple_dagger")


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
    VeryShort = 0
    Short = 1
    Long = 2
    VeryLong = 3


def clone_bc(venv, expert_trainer, student_model, duration, testing=True):
    env = venv.env

    # depending on what duration is, there are 2 different configs
    config_dict = {
        CloneDuration.VeryShort: {
            "min_episodes": 10,  # trajectories from start to done = True, as expert is deterministic, this is the same as n_env
            "evaluation_freq": 10,
            "n_batches": 5,
        },
        CloneDuration.Short: {
            "min_episodes": venv.env.n_envs,
            "evaluation_freq": 1000,
            "n_batches": 5000,
        },
        CloneDuration.Long: {
            "min_episodes": venv.env.n_envs,
            "evaluation_freq": 25_000,
            "n_batches": 50_0,
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

    if testing:
        n = 0
        all_val = []
        while True:
            try:
                obs = transitions.obs[n]
                act = transitions.acts[n]
                current_val = obs.tolist() + act.tolist()
                all_val.append(current_val)

                n += 1
            except:
                break

        pd.DataFrame(all_val).to_csv(
            "/Users/juusoahlroos/Documents/own/gradu/src/data_management/clone_data.csv",
            index=False,
        )

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
        wait=15,  # how long to wait until first evaluation
        patience=6,
        n_episodes=5,  # how many episodes to evaluate on
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


def compare_cloned(
    venv, model, expert_policy, expert_params, action_count=100, normalize=True
):
    venv.n_env = 1
    if type(model) == str:
        model = load_trained_model(model, venv, normalize=normalize)
    # first run 10 random observations and compare them to expert
    expert = expert_policy(env=venv.env, **expert_params)
    expert_func = expert.get_action_func()

    for i in range(10):
        obs = venv.observation_space.sample().reshape(1, -1)
        action_model, _states = model.predict(obs, deterministic=True)
        expert_action = expert_func(obs)
        print(f"Observation:    {obs}")
        print(f"Action:         {action_model}")
        print(f"Expert action:  {expert_action}\n")

    # then run 10 predictions against the expert and see if answer is always the same
    obs = venv.observation_space.sample().reshape(1, -1)
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
    obs = venv.reset()
    action_count = 10
    acts = 0
    every_nth = 10_000
    print("current step", venv.env.current_step)
    print("Actions from stepping")
    while True:
        try:
            # this should already be normalized
            n += 1
            action_model, _states = model.predict(obs, deterministic=True)
            action_expert = expert_func(obs)
            # print(action_model)
            # print(action_expert, "\n")
            if n % every_nth == 0:
                if venv.env.obs_space == LinearObservation:
                    print(f"step obs: {venv.env.obs_space.convert_to_readable(obs)}")
                else:
                    print(f"step obs: {obs}")
                print(f"action model: {action_model}")
                print(f"action expert: {action_expert} \n")
                acts += 1
                if acts >= action_count:
                    break
            obs, _, dones, _ = venv.step(action_expert)
            if np.all(dones):
                break

        except Exception as e:
            print(e)
            break
    print(n)
    print(f"distinct action share: {disctint_actions/n*100}%")


def model_cloning():
    # setup model for cloning here
    # parameters are set at the beginning of the function
    clone = False
    model_name = "clone_bc"
    n_env = 1
    normalize = False
    obs_space = LinearObservation(LinearObservationSpaces.SimpleLinearSpace)
    act_space = ActionSpace.NoSizeAction
    cloning_model = Cloning.BC
    cloning_duration = CloneDuration.Long
    expert_policy = ASPolicyVec
    expert_params = {
        "max_order_size": 5,
        "tick_size": 0.0001,
        "max_ticks": 10,
        "price_decimals": 4,
        "inventory_target": 0,
        "risk_aversion": 0.2,
        "order_size": 1,
        "obs_type": obs_space,
        "act_type": act_space,
    }

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
    expert = expert_policy(env=env, **expert_params)
    if clone:
        if cloning_model == Cloning.BC:
            model_name = clone_bc(venv, expert, student_model, cloning_duration)
    compare_cloned(
        venv,
        model_name,
        expert_policy,
        expert_params,
        action_count=3,
        normalize=normalize,
    )


from environments.util import setup_venv
from data_management import get_data_by_dates


def cloning_v2():
    data = get_data_by_dates("2021-12-21", days=1)
    venv = setup_venv(data=data, n_env=8)

    cloning_duration = CloneDuration.Short
    expert_policy = ASPolicyVec
    expert_params = {
        "max_order_size": 5,
        "tick_size": 0.0001,
        "max_ticks": 10,
        "price_decimals": 4,
        "inventory_target": 0,
        "risk_aversion": 0.2,
        "order_size": 1,
        "obs_type": venv.env.obs_space,
        "act_type": venv.env.act_space,
    }
    expert = expert_policy(env=venv.env, **expert_params)
    student_model = PPO("MlpPolicy", venv, verbose=1)

    model_name = clone_bc(venv, expert, student_model, cloning_duration)
    compare_cloned(
        venv,
        model_name,
        expert_policy,
        expert_params,
        action_count=10,
    )


if __name__ == "__main__":
    # clone_manual_policy_dagger()
    # clone_manual_policy_bc()
    # clone_always_same_policy_bc()
    # clone_simple_policy()
    # clone_simple_policy_dagger()
    # compare_cloned()
    # model_cloning()
    # test_normalized_env()
    cloning_v2()
