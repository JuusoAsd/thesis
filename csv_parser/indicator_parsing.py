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


def parse_indicators_v1():
    # Start by creating readers for raw data, both order book and trades
    # NOTE: at the moment only trade data + mid price is needed because indicators do not require OB
    # Use rust parser that parses trades + order book into trade + mid price for source data
    interim_path = os.getenv("INTERIM_PATH")
    target_path = os.getenv("INDICATOR_PATH")
    day = 1000 * 60 * 60 * 24
    minute = 1000 * 60
    # set update interval to 0 for most calculations, intensity is 99.5% of time so update it to every 30 minutes
    intensity_estimator = IntensityEstimator(lookback=day, update_interval=minute * 30)
    volatility_estimator = VolatilityEstimator(
        lookback=day, return_aggregation=minute, update_interval=0
    )
    osi_estimator = OSIEstimator(lookback=day)

    n = 0
    last_update = 0
    start_updates = 10000
    update_interval = minute
    with open(target_path, "+w") as f_target:
        writer = csv.writer(f_target)
        writer.writerow(["timestamp", "intensity", "volatility", "osi"])
        with open(interim_path, "r") as f:
            for line in f:
                n += 1
                line = line.rstrip().split(",")
                if n == 1:
                    continue
                timestamp = int(line[0])
                mid_price = float(line[1])
                size = float(line[2])
                price = float(line[3])

                intensity_estimator.update([(timestamp, price, size, mid_price)])
                volatility_estimator.update(mid_price, timestamp)
                osi_estimator.update([(timestamp, price, size, mid_price)])

                if start_updates < n and timestamp > last_update + update_interval:
                    start_calc = time.time()
                    _, intensity = intensity_estimator.get_value(timestamp)
                    intensity_seconds = time.time() - start_calc
                    volatility = volatility_estimator.get_value(timestamp)
                    volatility_seconds = time.time() - start_calc - intensity_seconds
                    osi = osi_estimator.get_value(timestamp)
                    osi_seconds = (
                        time.time()
                        - start_calc
                        - intensity_seconds
                        - volatility_seconds
                    )
                    writer.writerow([timestamp, intensity, volatility, osi])
                    last_update = timestamp

                    total_time = time.time() - start_calc

                    logging.info(
                        f"total time: {total_time}, intensity share: {round(intensity_seconds / total_time,4)}, volatility share: {round(volatility_seconds / total_time,4)}, osi share: {round(osi_seconds / total_time,4)}"
                    )

                if n % 100000 == 0:
                    print(f"Processed {n} lines")


if __name__ == "__main__":
    parse_indicators_v1()
