"""
Parses raw csv data into csv of indicators
"""
import os
from dotenv import load_dotenv
from AS.estimators import (
    IntensityEstimator,
    VolatilityEstimator,
    OSIEstimator,
)

load_dotenv("parse.env")
import csv
import time
import logging
from datetime import datetime, timezone

day = 1000 * 60 * 60 * 24
minute = 1000 * 60


class BaseReader:
    def __init__(self, path):
        self.path = path
        self.line_count = 0
        self.file = open(path, "r")
        self.file.readline()  # skip header
        self.read_line()

    def read_line(self):
        line = self.file.readline()
        if line == "":
            return False
        timestamp, bid, ask, low, high, _, _ = line.rstrip().split(",")
        mid = (float(bid) + float(ask)) / 2
        timestamp = int(timestamp)
        low = float(low)
        high = float(high)
        self.last_timestamp = timestamp
        self.last_mid = mid
        self.last_low = low
        self.last_high = high
        self.line_count += 1
        return True

    def get_last(self):
        return [self.last_timestamp, self.last_mid, self.last_low, self.last_high]


class IndicatorReader:
    def __init__(self, path):
        self.line_count = 0
        self.path = path
        self.file = open(path, "r")
        self.file.readline()  # skip header

        self.intensity_estimator = IntensityEstimator(
            lookback=day, update_interval=minute * 30
        )
        self.volatility_estimator = VolatilityEstimator(
            lookback=day, return_aggregation=minute, update_interval=1000
        )
        self.osi_estimator = OSIEstimator(lookback=day)
        self.read_line()

    def read_line(self):
        line = self.file.readline()
        if line == "":
            return False
        timestamp, mid, size, price, _ = line.rstrip().split(",")
        timestamp = int(timestamp)
        mid = round(float(mid), 5)
        size = float(size)
        price = float(price)
        self.last_timestamp = timestamp
        self.last_mid = mid
        self.last_size = size
        self.last_price = price

        self.intensity_estimator.update([(timestamp, price, size, mid)])
        self.volatility_estimator.update(mid, timestamp)
        self.osi_estimator.update([(timestamp, price, size, mid)])
        self.line_count += 1

        return True

    def get_last(self):
        return [
            self.intensity_estimator.get_value(self.last_timestamp)[1],
            self.volatility_estimator.get_value(self.last_timestamp),
            self.osi_estimator.get_value(self.last_timestamp),
        ]


def parse_indicators_v1():
    # Start by creating readers for raw data, both order book and trades
    # NOTE: at the moment only trade data + mid price is needed because indicators do not require OB
    # Use rust parser that parses trades + order book into trade + mid price for source data
    interim_path = os.getenv("INTERIM_PATH")
    base_path = os.getenv("BASE_PATH")
    target_path = os.getenv("INDICATOR_PATH")

    # set update interval to 0 for most calculations, intensity is 99.5% of time so update it to every 30 minutes

    n = 0
    last_update = 0
    start_updates = 10000
    update_interval = minute
    write_path = None
    current_date = None

    indicators = IndicatorReader(interim_path)
    base = BaseReader(base_path)

    # start by rewinding indicators by "start_updates"
    while n < start_updates:
        if not indicators.read_line():
            raise ValueError(f"Indicators done after {n} lines")
        n += 1

    while True:
        if indicators.last_timestamp < base.last_timestamp:
            if not indicators.read_line():
                print(f"Finished indicators")
                break
        else:
            if not base.read_line():
                print(f"Finished base")
                break
            # record values from both here
            current_date = datetime.fromtimestamp(
                base.last_timestamp / 1000, timezone.utc
            ).date()

            # initialize data file for new date
            if write_path == None or current_date != path_date:
                # close previous file
                if write_path != None:
                    f.close()
                path_date = current_date
                folder_path = os.path.join(
                    target_path, str(path_date).replace("-", "_")
                )
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                write_path = os.path.join(folder_path, "data.csv")
                f = open(write_path, "w+")
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "mid_price",
                        "low_price",
                        "high_price",
                        "intensity",
                        "volatility",
                        "osi",
                    ]
                )
            writer.writerow(base.get_last() + indicators.get_last())
            n += 1
            if n % 10000 == 0:
                print(
                    f"Lines processed, base: {base.line_count}, indicators: {indicators.line_count}"
                )

    print(
        f"Lines processed, base: {base.line_count}, indicators: {indicators.line_count}"
    )


if __name__ == "__main__":
    parse_indicators_v1()
