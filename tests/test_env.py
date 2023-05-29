import logging
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from src.environments.env_configs.policies import ASPolicyVec

from dotenv import load_dotenv

load_dotenv(".env")

from src.util import get_test_config
from src.environments.util import setup_venv_config, setup_venv
from src.environments.env_configs.spaces import (
    LinearObservationSpaces,
    LinearObservation,
)
from src.data_management import get_data_by_dates


config = get_test_config("test_env")


def test_trade_size_price():
    """
    Want to test that the trade size and price are applied correctly
    venv is set to use NormalizedAction
    """

    venv = setup_venv_config(config.data, config.env, config.venv)
    # assert type(venv.env.act_space) == type(ActionSpace.NormalizedAction)
    _ = venv.reset()
    action = np.array([1, 1, 1, 1])
    venv.env._apply_action(action)

    bid_size, ask_size, bid_price, ask_price, mid_price = (
        venv.env.bid_sizes,
        venv.env.ask_sizes,
        venv.env.bids,
        venv.env.asks,
        venv.env.mid_price[venv.env.current_step],
    )
    assert bid_size / venv.env.max_order_size == 1
    assert ask_size / venv.env.max_order_size == 1

    bid_ticks = (bid_price - mid_price) / venv.env.tick_size
    ask_ticks = (ask_price - mid_price) / venv.env.tick_size

    assert np.round(bid_ticks) / venv.env.max_order_size == 1
    assert np.round(ask_ticks) / venv.env.max_order_size == 1


test_data = {
    "best_bid": [99, 99],  # best bid price, used to determine market order
    "best_ask": [101, 101],
    "low_price": [50, 50],  # low price, used to determine limit buys
    "high_price": [150, 150],  # high price, used to determine limit sells
}
df = pd.DataFrame(test_data)
df["mid_price"] = (df["best_bid"] + df["best_ask"]) / 2


def test_limit_order(caplog):
    # caplog.set_level(logging.DEBUG)
    venv = setup_venv(
        df,
        obs_space=LinearObservation(
            LinearObservationSpaces.OnlyInventorySpace,
        ),
        env_params={"max_ticks": 30},
    )
    venv.env.reset_metrics_on_reset = False
    action = np.array([1, 1, -1, 1])
    obs, reward, done, info = venv.step(action)
    expected_value = (
        venv.env.capital + 30 * venv.env.tick_size * 2 * venv.env.max_order_size
    )
    values = venv.env.get_raw_recorded_values()
    # check if PnL is correct
    assert expected_value == values["values"][0][0], "limit order PnL"
    # check if inventory is correct
    assert venv.env.inventory_qty == 0, "limit order inventory"
    assert_allclose(venv.env.spread, [0.06])


def test_market_order(caplog):
    # caplog.set_level(logging.DEBUG)
    venv = setup_venv(
        df,
        obs_space=LinearObservation(
            LinearObservationSpaces.OnlyInventorySpace,
        ),
        env_params={"max_ticks": 30_000},
    )
    venv.reset()
    venv.env.reset_metrics()
    venv.env.reset_metrics_on_reset = False

    action = np.array([1, 1, 1, -1])

    obs, reward, done, info = venv.step(action)
    values = venv.env.get_raw_recorded_values()

    # expect to buy 10 at best ask (101) and sell 10 at best bid (99)
    expected_value = 10 * (99 - 101) + venv.env.capital
    assert (
        expected_value == values["values"][0][0]
    ), f"market order PnL, {expected_value} != {values['values'][0][0]}"
    # check if inventory is correct
    assert venv.env.inventory_qty == 0, "market order inventory"
    assert_allclose(venv.env.spread, [-60])


def test_zero_qty():
    venv = setup_venv(
        df,
        obs_space=LinearObservation(
            LinearObservationSpaces.OnlyInventorySpace,
        ),
        env_params={"max_ticks": 30},
    )
    venv.env.reset_metrics_on_reset = False

    action = np.array([0, 0, -1, 1])
    obs, reward, done, info = venv.step(action)
    expected_value = venv.env.capital

    values = venv.env.get_raw_recorded_values()
    # check if PnL is correct
    assert expected_value == values["values"][0][0], "no trades PnL"
    # check if inventory is correct
    assert venv.env.inventory_qty == 0, "no trades inventory"


def test_metrics():
    venv = setup_venv(
        df,
        obs_space=LinearObservation(
            LinearObservationSpaces.OnlyInventorySpace,
        ),
        env_params={"max_ticks": 30},
    )
    venv.env.reset_metrics_on_reset = False

    values = [100, 80, 110, 120]
    inventory = [0.5, 0.7, 0.9]
    venv.env.values = [np.array([x]) for x in values]
    venv.env.inventory_values = [np.array([x]) for x in inventory]

    metrics = venv.env.get_metrics()
    assert pytest.approx(metrics["episode_return"][0], rel=1e-5) == 0.2
    assert pytest.approx(metrics["drawdown"][0], rel=1e-5) == -0.2
    assert pytest.approx(metrics["max_inventory"][0], rel=1e-5) == 0.9
    assert pytest.approx(metrics["mean_abs_inv"][0], rel=1e-5) == np.mean(
        np.abs(np.array(inventory))
    )


# def test_metrics_parallel():
#     venv = setup_venv(
#         df,
#         obs_space=LinearObservation(
#             LinearObservationSpaces.OnlyInventorySpace,
#         ),
#         env_params={"max_ticks": 30},
#     )
#     values = [100, 80, 110, 120]
#     inventory = [0.5, 0.7, 0.9]
#     venv.env.values = [np.array([x, x]) for x in values]
#     venv.env.inventory_values = [np.array([x, x]) for x in inventory]
#     metrics = venv.env.get_metrics()
#     print(metrics)
#     sharpe = 0.2 / np.std(np.diff(np.array(values)))
#     inventory = np.mean(np.abs(np.array(inventory)))
#     assert pytest.approx(metrics["episode_return"][0], rel=1e-5) == 0.2
#     assert pytest.approx(metrics["episode_return"][1], rel=1e-5) == 0.2
#     assert pytest.approx(metrics["sharpe"][0], rel=1e-5) == sharpe
#     assert pytest.approx(metrics["sharpe"][1], rel=1e-5) == sharpe
#     assert pytest.approx(metrics["drawdown"][0], rel=1e-5) == -0.2
#     assert pytest.approx(metrics["drawdown"][1], rel=1e-5) == -0.2
#     assert pytest.approx(metrics["max_inventory"][0], rel=1e-5) == 0.9
#     assert pytest.approx(metrics["max_inventory"][1], rel=1e-5) == 0.9
#     assert pytest.approx(metrics["mean_abs_inv"][0], rel=1e-5) == inventory
#     assert pytest.approx(metrics["mean_abs_inv"][1], rel=1e-5) == inventory


"""
Create a test that clearly shows the structure of actions, observations, rewards and saved values
for both 1 env and multi env. assert that the values and shapes are correct
"""


def test_single_env():
    venv = setup_venv_config(config.data, config.env_single, config.venv)
    action = np.array([1, 1, -1, 1])
    obs, reward, done, info = venv.step(action)
    assert action.shape == (4,)
    assert obs.shape == (1, 3)
    assert reward.shape == (1,)
    assert done.shape == (1,)
    print(action, obs, reward, done)


def test_multi_env():
    venv = setup_venv_config(config.data, config.env_multi, config.venv)
    action_single = np.array([1, 1, -1, 1]).reshape(1, -1)
    actions = np.repeat(action_single, 5, axis=0)
    obs, reward, done, info = venv.step(actions)

    assert actions.shape == (5, 4)
    assert obs.shape == (5, 3)
    assert reward.shape == (5,)
    assert done.shape == (5,)


def test_everything_linear_as():
    venv = setup_venv_config(config.data, config.env_everything_linear_as, config.venv)
    obs = venv.reset()
    expert_policy = ASPolicyVec(env=venv.env, **config.expert_params)
    func = expert_policy.get_no_size_action_linear
    obs_expert = func(obs[:, [2, 3, 4]])
    assert obs_expert.tolist()[0] == obs.tolist()[0][:2]


def test_everything_linear_as_parallel():
    venv = setup_venv_config(
        config.data, config.env_everything_linear_as_parallel, config.venv
    )
    obs = venv.reset()
    expert_policy = ASPolicyVec(env=venv.env, **config.expert_params)
    func = expert_policy.get_no_size_action_linear
    obs_expert = func(obs[:, [2, 3, 4]])
    np.testing.assert_allclose(obs[:, [0, 1]], obs_expert, atol=0.001)
