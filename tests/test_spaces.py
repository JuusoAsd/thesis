"""
Most spaces are simple, only test the LinearObservation class
"""
import sys, os, logging
import numpy as np
import pandas as pd
import pytest

from dotenv import load_dotenv

load_dotenv(".env")

from src.data_management import get_data_by_dates
from src.util import get_test_config
from src.environments.util import setup_venv_config, setup_venv
from src.environments.env_configs.policies import ASPolicyVec
from src.environments.env_configs.spaces import (
    ActionSpace,
    LinearObservationSpaces,
    LinearObservation,
)
from src.environments.env_configs.spaces import (
    LinearObservation,
    LinearObservationSpaces,
)
import time


def test_setup():
    space = LinearObservation(LinearObservationSpaces.OnlyInventorySpace)
    assert space.obs_space.low == np.array([-1])
    assert space.obs_space.high == np.array([1])


def test_convert_ok():
    space = LinearObservation(LinearObservationSpaces.SimpleLinearSpace)
    obs_dict = {
        "inventory": np.array([1]),
        "volatility": np.array([0.01]),
        "intensity": np.array([50_000]),
    }
    res = space.convert_to_normalized([1, 0.01, 50_000])
    assert res.tolist() == [0.5, 1, 0]


def test_normalize_arrays():
    space = LinearObservation(LinearObservationSpaces.SimpleLinearSpace)
    line = np.array([1, 0.01, 50_000])
    arr = np.array([line, line, line])
    res = space.convert_to_normalized(arr)
    assert res.shape == (3, 3)
    assert res[0].tolist() == [0.5, 1, 0]
    assert res[1].tolist() == [0.5, 1, 0]


def test_normalize_arrays_min_max():
    space = LinearObservation(LinearObservationSpaces.SimpleLinearSpace)
    # convert everything to min values
    first = np.array([-3, -1, -1])
    # convert everything to max values
    second = np.array([3, 1, 1e8])
    # convert everything to middle values
    third = np.array([0, 0.005, 50_000])

    arr = np.array([first, second, third])
    res = space.convert_to_normalized(arr)
    assert res.shape == (3, 3)
    assert res[0].tolist() == [-1, -1, -1]
    assert res[1].tolist() == [1, 1, 1]
    assert res[2].tolist() == [0, 0, 0]


def test_denormalize_arrays_min_max():
    space = LinearObservation(LinearObservationSpaces.SimpleLinearSpace)

    # from min values
    first = np.array([-1, -1, -1])
    # from max values
    second = np.array([1, 1, 1])
    # from middle values
    third = np.array([0, 0, 0])

    arr = np.array([first, second, third])

    res = space.convert_to_readable(arr)
    assert res.shape == (3, 3)
    assert res[0].tolist() == [-2, 0, 0]
    assert res[1].tolist() == [2, 0.01, 100_000]
    assert res[2].tolist() == [0, 0.005, 50_000]


# def test_space_input():
#     space = LinearObservation(LinearObservationSpaces.SimpleLinearSpace)
