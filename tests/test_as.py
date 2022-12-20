import math
import logging
import numpy as np
from scipy.optimize import curve_fit
from csv_parser.AS.intensity import IntensityEstimator, curve_func
import pandas as pd


def test_read_line_trades(caplog):
    # TODO: currently midprice is not used but it should be included in calculating intensity (see update_trades)
    # TODO: use rust to parse trades and orderbook data so that it records following:
    # column 1, timestamps in milliseconds
    # column 2, current midprice based on orderbook
    # column 3, trade price
    # column 4: trade size
    # recording is done every 10 milliseconds OR whenever a trade happens
    caplog.set_level(logging.INFO)

    n = 0
    with open(
        "/home/juuso/Documents/gradu/parsed_data/trades/ADAUSDT-trades-2021-12-21.csv",
        "r",
    ) as f:
        main_estimator = IntensityEstimator()
        secondary_estimator = IntensityEstimator()
        full_estimator = IntensityEstimator()

        previous_time = 0
        lookback_start = 0

        # setting parameters, everything is in milliseconds:
        # record_interval, how many timestamps between each recorded value
        # lookback, how many timestamps to look back when estimating intensity
        # TODO: Smoothing? is there some moving average parameter? This should probably be done later...
        record_interval = 10_000
        lookback = 5_000_000
        lookback_half = lookback / 2

        # we start by updating main estimator with trades and recording the value every record_interval
        # after lookback_half, we also start updating secondary estimator with the same trades
        # after lookback, we set the secondary as main and reset secondary
        while True:
            n += 1
            line = f.readline().rstrip().split(",")
            price = float(line[1])
            amount = float(line[2])
            time = int(line[4])
            if lookback_start == 0:
                lookback_start = time

            if time >= previous_time + record_interval:
                main_estimator.estimate_intensity([(price, amount)])
                full_estimator.estimate_intensity([(price, amount)])
                previous_time = time
            else:
                main_estimator.update_trades([(price, amount)])
                full_estimator.update_trades([(price, amount)])

            if time >= lookback_start + lookback_half:
                secondary_estimator.update_trades([(price, amount)])

            if time >= lookback_start + lookback:
                logging.info(
                    f"Switching estimators at {time} with {main_estimator.trade_count} trades"
                )
                main_estimator = secondary_estimator
                secondary_estimator = IntensityEstimator()
                lookback_start = time


def test_hummingbot(caplog):
    # Testing estimate calculations as per: https://github.com/hummingbot/hummingbot/blob/433df629f3a486a1b6e83bfd8a19493ba20e7e85/test/hummingbot/strategy/utils/trailing_indicators/test_trading_intensity.py#L230
    caplog.set_level(logging.DEBUG)

    estimator = IntensityEstimator()
    last_price = 1
    trade_price_levels = [2, 3, 4, 5]

    a = 2
    b = 0.1

    ts = [curve_func(p - last_price, a, b) for p in trade_price_levels]

    timestamp = pd.Timestamp("2019-01-01", tz="UTC").timestamp()
    timestamp += 1

    estimator.update_trades(zip(trade_price_levels, ts, [last_price] * 4))
    estimator.calculate_current_values()

    alpha = math.isclose(estimator.alpha, a, rel_tol=1e-8)
    kappa = math.isclose(estimator.kappa, b, rel_tol=1e-8)

    assert alpha, f"{estimator.alpha / a * 1e10} is not {1e10}"
    assert kappa, f"{estimator.kappa / b * 1e10} is not {1e10}"
