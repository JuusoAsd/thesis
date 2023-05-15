import logging
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from dotenv import load_dotenv

load_dotenv(".env")

from src.util import get_test_config
from src.environments.util import setup_venv_config, setup_venv
from src.environments.env_configs.spaces import (
    LinearObservationSpaces,
    LinearObservation,
)


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
    values = [100, 80, 110, 120]
    inventory = [0.5, 0.7, 0.9]
    venv.env.values = [np.array([x]) for x in values]
    venv.env.inventory_values = [np.array([x]) for x in inventory]

    metrics = venv.env.get_metrics()
    assert pytest.approx(metrics["episode_return"][0], rel=1e-5) == 0.2
    assert pytest.approx(metrics["sharpe"][0], rel=1e-5) == 0.2 / np.std(
        np.diff(np.array(values))
    )
    assert pytest.approx(metrics["drawdown"][0], rel=1e-5) == -0.2
    assert pytest.approx(metrics["max_inventory"][0], rel=1e-5) == 0.9
    assert pytest.approx(metrics["mean_abs_inv"][0], rel=1e-5) == np.mean(
        np.abs(np.array(inventory))
    )


def test_metrics_parallel():
    venv = setup_venv(
        df,
        obs_space=LinearObservation(
            LinearObservationSpaces.OnlyInventorySpace,
        ),
        env_params={"max_ticks": 30},
    )
    values = [100, 80, 110, 120]
    inventory = [0.5, 0.7, 0.9]
    venv.env.values = [np.array([x, x]) for x in values]
    venv.env.inventory_values = [np.array([x, x]) for x in inventory]
    metrics = venv.env.get_metrics()

    print(metrics)
    liquidated = metrics["max_inventory"] > 0.5
    print(liquidated)
    val = metrics["sharpe"] * (1 - liquidated) + liquidated * 100
    print(metrics["sharpe"])
    print(val)
    val_conc = np.array([metrics["sharpe"], val])
    print(val_conc)
    print(np.min(val_conc, axis=0))

    sharpe = 0.2 / np.std(np.diff(np.array(values)))
    inventory = np.mean(np.abs(np.array(inventory)))
    assert pytest.approx(metrics["episode_return"][0], rel=1e-5) == 0.2
    assert pytest.approx(metrics["episode_return"][1], rel=1e-5) == 0.2
    assert pytest.approx(metrics["sharpe"][0], rel=1e-5) == sharpe
    assert pytest.approx(metrics["sharpe"][1], rel=1e-5) == sharpe
    assert pytest.approx(metrics["drawdown"][0], rel=1e-5) == -0.2
    assert pytest.approx(metrics["drawdown"][1], rel=1e-5) == -0.2
    assert pytest.approx(metrics["max_inventory"][0], rel=1e-5) == 0.9
    assert pytest.approx(metrics["max_inventory"][1], rel=1e-5) == 0.9
    assert pytest.approx(metrics["mean_abs_inv"][0], rel=1e-5) == inventory
    assert pytest.approx(metrics["mean_abs_inv"][1], rel=1e-5) == inventory
