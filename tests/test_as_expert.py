import sys, os, logging
import numpy as np

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/../src"))

from dotenv import load_dotenv

load_dotenv(".env")

from src.data_management import get_data_by_dates
from src.util import get_test_config

from src.environments.util import setup_venv_config
from src.environments.env_configs.policies import ASPolicyVec

config = get_test_config("test_as_expert")


def test_get_data_imports():
    data = get_data_by_dates("2021_12_30", "2021_12_31")
    assert data is not None


def test_create_try_expert(caplog):
    # caplog.set_level(logging.DEBUG)
    venv = setup_venv_config(config.data, config.env, config.venv)
    expert_policy = ASPolicyVec(venv.env, **config.expert_params)
    action_func = expert_policy.get_action_func()

    obs = venv.reset()
    for i in range(10):
        action = action_func(obs)
        obs, reward, done, info = venv.step(action)

    values = venv.env.get_raw_recorded_values()
    expected_inv = [-2, -2, 0, 0, -2, 0, -2, -2, 0, 0]
    assert values["inventory_qty"] == [np.array([x]) for x in expected_inv]


def test_create_try_expert_parallel(caplog):
    # caplog.set_level(logging.DEBUG)
    venv = setup_venv_config(config.data, config.env_parallel, config.venv)
    expert_policy = ASPolicyVec(venv.env, **config.expert_params)
    action_func = expert_policy.get_action_func()

    obs = venv.reset()
    for i in range(10):
        action = action_func(obs)
        obs, reward, done, info = venv.step(action)


def test_run_2_expert_parallel(caplog):
    # caplog.set_level(logging.DEBUG)
    venv = setup_venv_config(config.data, config.test_2_env, config.venv)
    expert_policy = ASPolicyVec(venv.env, **config.expert_params)
    action_func = expert_policy.get_action_func()

    obs = venv.reset()
    assert len(obs) == 2
    for i in range(10):
        action = action_func(obs)
        action2 = action.copy()
        full_action = np.array([action[0], action2[0]])
        obs, reward, done, info = venv.step(full_action)
        assert len(obs) == 2
        assert obs[0].tolist() == obs[1].tolist()

    values = venv.env.get_raw_recorded_values()
    expected_inv = [-2, -2, 0, 0, -2, 0, -2, -2, 0, 0]
    expected_arrays = [np.array([x, x]) for x in expected_inv]

    for i in range(len(values["inventory_qty"])):
        assert values["inventory_qty"][i].tolist() == expected_arrays[i].tolist()


def test_single_env_as():
    venv = setup_venv_config(config.data, config.env_single_as, config.venv)
    expert_policy = ASPolicyVec(venv.env, **config.expert_params)
    action_func = expert_policy.get_action_func()

    obs = venv.reset()
    for i in range(10):
        action = action_func(obs)
        obs, reward, done, info = venv.step(action)

    # print(obs, reward, done, info)
    assert action.shape == (1, 4)
    assert obs.shape == (1, 3)
    assert len(reward) == 1
    assert len(done) == 1
    assert len(info) == 1

    recorded = venv.env.get_raw_recorded_values()


def test_multi_env_as():
    venv = setup_venv_config(config.data, config.env_multi_as, config.venv)
    expert_policy = ASPolicyVec(venv.env, **config.expert_params)
    action_func = expert_policy.get_action_func()

    obs = venv.reset()
    for i in range(10):
        action = action_func(obs)
        obs, reward, done, info = venv.step(action)

    assert action.shape == (25, 4)
    assert obs.shape == (25, 3)
    assert len(reward) == 25
    assert len(done) == 25
    assert len(info) == 25

    metrics = venv.env.get_metrics()
    assert metrics["timesteps"] == 10
    assert metrics["episode_return"].shape == (25,)
    assert metrics["sharpe"].shape == (25,)
    assert metrics["drawdown"].shape == (25,)
    assert metrics["max_inventory"].shape == (25, 1)
    assert metrics["mean_abs_inv"].shape == (25, 1)

    recorded = venv.env.get_raw_recorded_values()
    first_val = recorded["values"][0][0]
    last_val = recorded["values"][-1][0]
    returns = last_val / first_val - 1
    assert metrics["episode_return"][0] == returns
