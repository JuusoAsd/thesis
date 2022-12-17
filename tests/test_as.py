import logging
import numpy as np
from scipy.optimize import curve_fit
from csv_parser.AS.intensity import IntensityEstimator, curve_func


def test_read_line_trades(caplog):
    caplog.set_level(logging.INFO)

    n = 0
    with open(
        "/home/juuso/Documents/gradu/parsed_data/trades/ADAUSDT-trades-2021-12-21.csv",
        "r",
    ) as f:
        main_estimator = IntensityEstimator()
        secondary_estimator = IntensityEstimator()

        previous_time = 0
        lookback_start = 0

        # setting parameters, everything is in milliseconds:
        # record_interval, how many timestamps between each recorded value
        # lookback, how many timestamps to look back when estimating intensity
        # TODO: Smoothing? is there some moving average parameter? This should probably be done later...
        record_interval = 100
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
                print(time, main_estimator.alpha, main_estimator.kappa)
                previous_time = time
            else:
                main_estimator.update_trades([(price, amount)])

            if time >= lookback_start + lookback_half:
                secondary_estimator.update_trades([(price, amount)])

            if time >= lookback_start + lookback:
                logging.info(
                    f"Switching estimators at {time} with {main_estimator.trade_count} trades"
                )
                main_estimator = secondary_estimator
                secondary_estimator = IntensityEstimator()
                lookback_start = time


def test_estimate():
    data = {
        1.2373: 10400.0,
        1.2372: 31860.0,
        1.2371: 35555.0,
        1.237: 28334.0,
        1.2369: 24401.0,
        1.2368: 10640.0,
        1.2367: 2343.0,
        1.2366: 3580.0,
        1.2365: 13150.0,
        1.2374: 25631.0,
        1.2375: 19264.0,
        1.2376: 31467.0,
        1.2378: 32398.0,
        1.2379: 22347.0,
        1.238: 45018.0,
        1.2381: 35458.0,
        1.2382: 43397.0,
        1.2383: 31523.0,
        1.2384: 31197.0,
        1.2385: 14862.0,
        1.2387: 10809.0,
        1.2388: 2630.0,
        1.2386: 15867.0,
        1.2377: 23174.0,
        1.2364: 3936.0,
        1.2363: 103.0,
        1.2362: 92.0,
        1.2361: 37.0,
        1.236: 97.0,
        1.2359: 2327.0,
        1.2358: 7025.0,
        1.2357: 8666.0,
        1.2356: 8649.0,
        1.2355: 27289.0,
        1.2354: 1070.0,
    }

    price_levels = np.array(list(data.keys()))
    price_levels.sort()
    price_levels = price_levels[::-1]
    lambdas = np.array([data[x] for x in price_levels])
    param, _ = curve_fit(
        f=curve_func,
        xdata=price_levels,
        ydata=lambdas,
        p0=(1, 1),
        method="dogbox",
        bounds=([0, 0], [np.inf, np.inf]),
    )
    print(param)
