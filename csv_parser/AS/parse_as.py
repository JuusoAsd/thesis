import csv
from src.environments.util import FileManager
from csv_parser.AS.estimators import IntensityEstimator
from csv_parser.AS.estimators import VolatilityEstimator
import logging


def parse_as_full():
    
    """
    Parse AS agent data and save it to a single CSV file / folder of files
    PARSED DATA:
        timestamp, best bid, best ask, (trade price/0, trade size/0), current vol estimate, current intensity
    """
    as_files = FileManager(r"C:\Users\Ville\Documents\gradu\parsed_data\AS\data.csv", headers=True)
    target_file = open(r"C:\Users\Ville\Documents\gradu\parsed_data\AS\data_full.csv", "w+", newline="")
    columns = [
        "timestamp",
        "best_bid",
        "best_ask",
        "trade_price",
        "trade_size",
        "vol_estimate",
        "intensity_estimate",
    ]
    start_period = 1000_000
    start_ts = 0
    current_ts = 0
    count = 0
    writer = csv.writer(target_file, delimiter=",")
    writer.writerow(columns)
    current_state = as_files.get_next_event()
    intensity = IntensityEstimator(lookback=250_000)
    volatility = VolatilityEstimator(lookback=1_000_000, return_aggregation=5_000)

    while current_state is not None:
        count += 1
        ts = int(current_state[0])
        if start_ts == 0:
            start_ts = ts
        mid_price = float(current_state[1])
        best_bid = float(current_state[2])
        best_ask = float(current_state[3])
        trade_size = float(current_state[4])
        trade_price = float(current_state[5])

        if trade_size != 0:
            intensity.update_trades([(ts, trade_price, trade_size, mid_price)])
        volatility.update_prices(mid_price, ts)

        if ts - start_ts > start_period:
            _, intensity_estimate = intensity.get_current_estimate(ts)
            vol_estimate = volatility.get_current_estimate(ts)

            writer.writerow(
                [
                    ts,
                    best_bid,
                    best_ask,
                    trade_price,
                    trade_size,
                    vol_estimate,
                    intensity_estimate,
                ]
            )
        current_state = as_files.get_next_event()
        if count % 100_000 == 0:
            print(f"Processed {count} lines")
