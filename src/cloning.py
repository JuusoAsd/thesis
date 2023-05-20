import os
import logging
from enum import Enum

import pickle


from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from stable_baselines3 import PPO
import torch as th
from imitation.algorithms import bc
from imitation.data import rollout
from stable_baselines3.common.vec_env import VecNormalize


from src.environments.env_configs.policies import ASPolicyVec
from src.environments.mm_env_vec import MMVecEnv, SBMMVecEnv
from src.environments.env_configs.rewards import *
from src.environments.env_configs.spaces import (
    ActionSpace,
    LinearObservationSpaces,
    LinearObservation,
)

from src.model_testing import trained_vs_manual
from src.data_management import get_data_by_dates
from src.util import get_model_hash
from src.util import get_config, create_config_hash
from src.environments.util import setup_venv_config

load_dotenv()


def load_transitions(name):
    try:
        dir = Path(f"{os.getenv('COMMON_PATH')}/models/transitions")
        with open(dir / f"{name}.pickle", "rb") as f:
            transitions = pickle.load(f)
            logging.info(f"loaded transitions")
            return transitions
    except:
        return None


def save_transitions(transitions, name):
    dir = Path(f"{os.getenv('COMMON_PATH')}/models/transitions")
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir / f"{name}.pickle", "wb") as f:
        pickle.dump(transitions, f)
    logging.info(f"saved transitions")


def save_model_by_config(config, model):
    """
    Save the model based on the config. Must be unique to that it can be reused.
    """
    hash = get_model_hash(config)
    dir = Path(f"{os.getenv('COMMON_PATH')}/models/{hash}")
    model.save(dir)


def load_model_by_config_hash(config, venv):
    hash = get_model_hash(config)
    if not check_model_exists_by_config(config):
        raise Exception(f"Model does not exist for {hash}")
    path = Path(f"{os.getenv('COMMON_PATH')}/models/{hash}.zip")
    model = PPO.load(path, env=venv)
    return model


def load_model_by_config(config, venv):
    # .zip is not appended to the path here for some reason????
    path = Path(
        f"{os.getenv('COMMON_PATH')}/models/{config.model.model_name}"
    )
    model = PPO.load(path, env=venv)
    return model


def check_model_exists_by_config(config):
    hash = get_model_hash(config)
    path = Path(f"{os.getenv('COMMON_PATH')}/models/{hash}.zip")
    return os.path.exists(path)


def get_transitions(config, venv):
    """
    Creates or loads the transitions used for cloning.
    Most of the items are fixed, however only observation and action space affect the transitions
    Use those for saving/loading
    TODO: maybe something else also affect the transitions?
    - Actual number of envs does, therefore also how the venvs are setup but that is ok
    """
    logging.info(f"Getting transitions")
    expert_policy = ASPolicyVec
    expert = expert_policy(env=venv.env, **config.expert_params)
    rng = np.random.default_rng(0)
    name = f"{config.env.spaces.action_space}_{config.env.spaces.observation_space.type}_transitions"
    loaded = load_transitions(name)
    if loaded is None:
        rollouts = rollout.rollout(
            expert.get_action_func(),
            venv,
            rollout.make_sample_until(min_timesteps=None, min_episodes=venv.env.n_envs),
            rng=rng,
            unwrap=False,
            verbose=True,
        )
        transitions = rollout.flatten_trajectories(rollouts)
        save_transitions(transitions, name)
    else:
        transitions = loaded

    return transitions


def clone_expert_config(config, venv, student_model, transitions):
    logging.info(f"cloning")
    # start by checking if this model already exists
    if check_model_exists_by_config(config):
        return
    student_policy = student_model.policy
    rng = np.random.default_rng(0)
    imitation_trainer = bc.BC(
        observation_space=venv.env.observation_space,
        action_space=venv.env.action_space,
        policy=student_policy,  # USE SAME ARCHITECTURE AS MODEL!!!
        demonstrations=transitions,
        batch_size=32,
        optimizer_cls=th.optim.Adam,
        optimizer_kwargs=dict(lr=0.001),
        ent_weight=1e-3,
        l2_weight=0.0,
        rng=rng,
    )
    tries = 3
    while True:
        # train with 5k batches = 2mins
        imitation_trainer.train(
            n_batches=5_000,
            log_interval=10_000,
            log_rollouts_venv=venv,
            log_rollouts_n_episodes=1,
            progress_bar=False
            # on_batch_end=eval_callback,
        )
        model_res, expert_res = trained_vs_manual(venv.clone_venv(), student_model)
        if (
            expert_res["trades"] * (1 - config.cloning.tolerance)
            < model_res["trades"]
            < expert_res["trades"] * (1 + config.cloning.tolerance)
        ):
            break
        tries -= 1
        if tries == 0:
            raise Exception("Could not clone model")

    # if cloning was succesful, save the model
    save_model_by_config(config, student_model)


def save_trained_model(model, path):
    """
    Save trained model optionally with its replay buffer
    and ``VecNormalize`` statistics

    model: the trained model
    dir: optional sub-dir of self.save_path to save the model in
    """
    model_dir = Path(f"{os.getenv('COMMON_PATH')}/models")

    model_path = str(model_dir / path)
    model.save(model_path)

    wrapper = model.get_vec_normalize_env()
    if wrapper is not None:
        wrapper.save(str(model_dir / (path + "_vec_normalize.pkl")))


def load_trained_model(name, venv, normalize=True, model_kwargs={}):
    model_dict = {}
    model_dir = Path(f"{os.getenv('COMMON_PATH')}/models")

    model_path = str(model_dir / name)
    normalize_path = str(model_dir / (name + "_vec_normalize.pkl"))
    if os.path.exists(normalize_path) and normalize:
        normalized_venv = VecNormalize.load(normalize_path, venv)
        model = PPO.load(model_path, env=normalized_venv, **model_kwargs)
    else:
        model = PPO.load(path=model_path, env=venv, **model_kwargs)

    return model


class Cloning(Enum):
    BC = 1
    Dagger = 2


class CloneDuration(Enum):
    VeryShort = 0
    Short = 1
    Long = 2
    VeryLong = 3


def clone_bc(venv, expert_trainer, student_model, duration, model_name, testing=True):
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
    rng = np.random.default_rng(0)

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

    imitation_trainer.train(
        n_batches=chosen_config["n_batches"],
        log_interval=10_000,
        log_rollouts_venv=venv,
        log_rollouts_n_episodes=1,
        # on_batch_end=eval_callback,
    )
    save_trained_model(student_model, model_name)
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


def cloning_v2():
    config = get_config("clone_config")

    venv = setup_venv_config(config.data, config.env, config.venv)

    cloning_duration = CloneDuration.Short
    expert_policy = ASPolicyVec
    expert = expert_policy(env=venv.env, **config.expert_params)
    student_model = PPO(
        "MlpPolicy", venv, verbose=1, policy_kwargs=config.policy_kwargs
    )

    model_name = clone_bc(
        venv,
        expert,
        student_model,
        cloning_duration,
        testing=False,
        model_name="clone_large",
    )


def cloning_multiple_envs():
    config = get_config("clone_config_multiple")
    venv = setup_venv_config(config.clone_data, config.env, config.venv)
    cloning_duration = CloneDuration.Short
    expert_policy = ASPolicyVec
    expert = expert_policy(env=venv.env, **config.expert_params)

    for i in config.policy_kwargs:
        start_time = time.time()
        hash = create_config_hash(i)
        print(f"value: {i}, hash: {hash}")
        config_to_dict = OmegaConf.to_container(i, resolve=True)
        student_model = PPO("MlpPolicy", venv, verbose=1, policy_kwargs=config_to_dict)
        model_name = clone_bc(
            venv,
            expert,
            student_model,
            cloning_duration,
            testing=False,
            model_name=hash,
        )
        model = load_trained_model(hash, venv, False)
        single_venv = venv.clone_venv(
            get_data_by_dates(**config.verify_cloning_data).to_numpy()
        )
        trained_vs_manual(single_venv, model, False)
        print(f"DONE, took {round((time.time() - start_time)/60,2)} minutes\n")


if __name__ == "__main__":
    # clone_manual_policy_dagger()
    # clone_manual_policy_bc()
    # clone_always_same_policy_bc()
    # clone_simple_policy()
    # clone_simple_policy_dagger()
    # compare_cloned()
    # model_cloning()
    # test_normalized_env()
    # cloning_v2()
    cloning_multiple_envs()
