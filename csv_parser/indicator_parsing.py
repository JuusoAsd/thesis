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

    def read_line(self, update=True):
        try:
            line = self.file.readline()
            if line == "":
                return False
            self.line_count += 1
            timestamp, bid, ask, low, high, _, _, imbalance = line.rstrip().split(",")
            bid = float(bid)
            ask = float(ask)
            mid = round((bid + ask) / 2, 5)
            timestamp = int(timestamp)
            low = float(low)
            high = float(high)
            imbalance = float(imbalance)

            self.last_timestamp = timestamp
            if update:
                self.last_bid = bid
                self.last_ask = ask
                self.last_mid = mid
                self.last_low = low
                self.last_high = high
                self.last_imbalance = imbalance

                # using self.last_timestamp (timestamp in milliseconds), calculate current second, minute, hour
                self.last_second = int(timestamp / 1000) % 60
                self.last_minute = int(timestamp / 1000 / 60) % 60
                self.last_hour = int(timestamp / 1000 / 60 / 60) % 24
            return True

        except Exception as e:
            print(e)
            print(line, self.line_count)
            raise e

    def get_last(self):
        return [
            self.last_timestamp,
            self.last_bid,
            self.last_ask,
            self.last_mid,
            self.last_low,
            self.last_high,
            self.last_imbalance,
            self.last_second,
            self.last_minute,
            self.last_hour,
        ]


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

    def read_line(self, update=True):
        line = self.file.readline()
        self.line_count += 1

        try:
            if line == "":
                return False
            timestamp, mid, size, price, _ = line.rstrip().split(",")
            timestamp = int(timestamp)
            mid = round(float(mid), 5)
            size = float(size)
            price = float(price)
            self.last_timestamp = timestamp
            if update:
                self.last_mid = mid
                self.last_size = size
                self.last_price = price

                self.intensity_estimator.update([(timestamp, price, size, mid)])
                self.volatility_estimator.update(mid, timestamp)
                self.osi_estimator.update([(timestamp, price, size, mid)])

            return True
        except Exception as e:
            print(e)
            print(line, self.line_count)
            raise e

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
    redo = True

    n = 0
    last_update = 0
    start_updates = 10000
    update_interval = minute
    write_path = None
    current_date = None

    indicators = IndicatorReader(interim_path)
    base = BaseReader(base_path)

    if redo:
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
                            "best_bid",
                            "best_ask",
                            "mid_price",
                            "low_price",
                            "high_price",
                            "order_book_imbalance",
                            "current_second",
                            "current_minute",
                            "current_hour",
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
    else:
        # start by finding the latest date that has been processed
        last_date = None
        content = os.listdir(target_path)
        for i in content:
            if os.path.isdir(os.path.join(target_path, i)):
                current_date = datetime.strptime(i, "%Y_%m_%d").date()
                if last_date == None or current_date > last_date:
                    last_date = current_date

        if last_date == None:
            raise ValueError("No data found in target path")

        # read the data file for last date and get the last timestamp
        last_path = os.path.join(
            target_path, str(last_date).replace("-", "_"), "data.csv"
        )
        with open(last_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                pass
            last_timestamp = int(row[0])

        # get some legroom to start from
        start_ts = last_timestamp - 10_000_000

        # start looping the reader, when reach start ts, start updating the indicators, when reach last ts, start writing
        init = False
        while True:
            if not init:
                if (
                    base.last_timestamp >= start_ts
                    and indicators.last_timestamp >= start_ts
                ):
                    init = True
                    print(f"Data rewinded")
                elif base.last_timestamp < start_ts:
                    base.read_line(update=False)
                elif indicators.last_timestamp < start_ts:
                    indicators.read_line(update=False)

            else:
                if indicators.last_timestamp < base.last_timestamp:
                    if not indicators.read_line():
                        print(f"Finished indicators")
                        break
                else:
                    if not base.read_line():
                        print(f"Finished base")
                        break

                    if base.last_timestamp >= last_timestamp:
                        # record values from both here
                        current_date = datetime.fromtimestamp(
                            base.last_timestamp / 1000, timezone.utc
                        ).date()
                        if current_date <= last_date:
                            continue
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
                                    "best_bid",
                                    "best_ask",
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


def fix_files():
    interim_path = os.getenv("INTERIM_PATH")
    base_path = os.getenv("BASE_PATH")

    num_fields = None
    with open(interim_path, "r", newline="") as f_in, open(
        "/Volumes/ssd/gradu_data/parsed/interim_data_fix.csv", "w", newline=""
    ) as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        if num_fields == None:
            num_fields = len(next(reader))
        for row in reader:
            if len(row) == num_fields:
                writer.writerow(row)
            else:
                # Do nothing (skip this row)
                pass

    num_fields = None
    with open(base_path, "r", newline="") as f_in, open(
        "/Volumes/ssd/gradu_data/parsed/base_data_fix.csv", "w", newline=""
    ) as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        if num_fields == None:
            num_fields = len(next(reader))
        for row in reader:
            if len(row) == num_fields:
                writer.writerow(row)
            else:
                # Do nothing (skip this row)
                pass


if __name__ == "__main__":
    parse_indicators_v1()
    # fix_files()
