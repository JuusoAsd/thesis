import sys, os, logging
import numpy as np
import pandas as pd

from dotenv import load_dotenv

load_dotenv(".env")
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/../src"))

from src.data_management import get_data_by_dates
from src.util import get_test_config
from src.environments.util import setup_venv_config, setup_venv
from src.environments.env_configs.policies import ASPolicyVec
from src.environments.env_configs.spaces import (
    ActionSpace,
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


def test_limit_order():
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


def test_market_order():
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
