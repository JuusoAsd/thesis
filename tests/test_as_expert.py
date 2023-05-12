import sys, os
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
    venv = setup_venv_config(config.data, config.env_parallel, config.venv)
    expert_policy = ASPolicyVec(venv.env, **config.expert_params)
    action_func = expert_policy.get_action_func()

    obs = venv.reset()
    for i in range(10):
        action = action_func(obs)
        obs, reward, done, info = venv.step(action)


def test_run_2_expert_parallel():
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
