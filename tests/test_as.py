import math
import logging
from csv_parser.AS.estimators import IntensityEstimator, curve_func
from csv_parser.AS.estimators import VolatilityEstimator
import pandas as pd


def test_read_line_trades(caplog):
    caplog.set_level(logging.INFO)

    n = 0
    filepath = "./parsed_data/AvellanedaStoikov/data_reverse.csv"
    with open(
        filepath,
        "r",
    ) as f:

        columns = f.readline().rstrip().split(",")
        col = {k: n for n, k in enumerate(columns)}

        # lookback in milliseconds
        main_estimator = IntensityEstimator(lookback=250_000)
        volatility_estimate = VolatilityEstimator()

        while True:
            n += 1
            line = f.readline()
            if not line:
                print(f"End of file reached after {n} lines")
                break
            line = line.rstrip().split(",")
            size = float(line[col["size"]])
            if size == 0:
                continue

            time = int(line[col["timestamp"]])
            mid_price = float(line[col["mid_price"]])
            trade_price = float(line[col["price"]])

            # always update the estimators with trades
            main_estimator.update_trades([(time, trade_price, size, mid_price)])
            volatility_estimate.update_prices(mid_price)

            if n % 100_000 == 0:
                main_estimator.calculate_current_values()
                print(f"Processed {n} lines, kappa: {main_estimator.kappa}")
                # print(f"{main_estimator.trades}")


def test_hummingbot(caplog):
    # Testing estimate calculations as per: https://github.com/hummingbot/hummingbot/blob/433df629f3a486a1b6e83bfd8a19493ba20e7e85/test/hummingbot/strategy/utils/trailing_indicators/test_trading_intensity.py#L230
    caplog.set_level(logging.DEBUG)

    estimator = IntensityEstimator(lookback=1000)
    last_price = 1
    trade_price_levels = [2, 3, 4, 5]

    a = 2
    b = 0.1

    size = [curve_func(p - last_price, a, b) for p in trade_price_levels]

    timestamp = pd.Timestamp("2019-01-01", tz="UTC").timestamp()
    timestamp += 1

    estimator.update_trades(zip([1] * 4, trade_price_levels, size, [last_price] * 4))
    estimator.calculate_current_values()

    alpha = math.isclose(estimator.alpha, a, rel_tol=1e-8)
    kappa = math.isclose(estimator.kappa, b, rel_tol=1e-8)

    assert alpha, f"{estimator.alpha / a * 1e10} is not {1e10}"
    assert kappa, f"{estimator.kappa / b * 1e10} is not {1e10}"


def test_as_orderbook():
    filepath = "./parsed_data/orderbook/2021-12-21.csv"
    every_n = 1000
    every_write = 1_000_00
    n = 0
    main_estimator = IntensityEstimator()
    with open(
        filepath,
        "r",
    ) as f:
        while True:
            if n % every_n == 0:
                line = f.readline()
                line = line.rstrip().split(",")
                mid_price = float(line[1])
                price_next = True
                for j in line[2:]:
                    if price_next:
                        price = float(j)
                        price_next = False
                    else:
                        size = float(j)
                        price_next = True
                        main_estimator.update_trades([(price, size, mid_price)])
            if n % every_write == 0:
                main_estimator.calculate_current_values()
                print(f"Processed {n} lines, kappa: {main_estimator.kappa}")
                print(main_estimator.trades)

            n += 1


def test_parse_full():
    from csv_parser.AS.parse_as import parse_as_full

    parse_as_full()


from src.environments.as_agent import ASAgent
from src.environments.mm_env import MMEnv


def test_env(caplog):
    caplog.set_level(logging.DEBUG)
    target = "./parsed_data/AvellanedaStoikov/AS_full.csv"
    agent_params = {"risk_aversion": 0.1}
    env = MMEnv(target, ASAgent, agent_parameters=agent_params, price_decimals=4)
    env.reset()
    for i in range(1000):
        env.step(None)

    print(
        f"Cash: {env.quote_asset}, inventory: {env.base_asset}, value: {env.get_current_value()}"
    )
    exit()


def test_env_full(caplog):
    caplog.set_level(logging.INFO)
    target = "./parsed_data/AvellanedaStoikov/AS_full.csv"
    agent_params = {"risk_aversion": 0.1}
    env = MMEnv(target, ASAgent, agent_parameters=agent_params, price_decimals=4)
    env.reset()
    while True:
        try:
            env.step(None)
        except Exception as e:
            print(e)
            break
    print(
        f"Cash: {env.quote_asset}, inventory: {env.base_asset}, value: {env.get_current_value()}"
    )
    exit()
