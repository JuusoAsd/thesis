import logging
import numpy as np
from scipy.optimize import curve_fit


def curve_func(t, a, b):
    return a * np.exp(-b * t)


# you can run this by running python -m pytest tests/test_as.py
class IntensityEstimator:
    def __init__(self):
        self.trades = {}
        self.kappa = 2
        self.alpha = 2
        self.trade_count = 0

    def update_trades(self, new_trades):
        # new trades is an array containing [trade price, trade amount and current mid price when trade took place]
        logging.debug(f"new trades: {new_trades}")
        for trade_price, amount, mid_price in new_trades:
            self.trade_count += 1
            price_diff = round(abs(trade_price - mid_price), 5)
            if price_diff in self.trades:
                self.trades[price_diff] += amount
            else:
                self.trades[price_diff] = amount

    def calculate_current_values(self):
        price_levels = np.array(list(self.trades.keys()))
        price_levels.sort()
        # reverse price_levels
        price_levels = price_levels[::-1]

        lambdas = []
        for i in price_levels:
            lambdas.append(self.trades[i])
        lambdas = np.array(lambdas)
        param, _ = curve_fit(
            f=curve_func,
            xdata=price_levels,
            ydata=lambdas,
            p0=(self.alpha, self.kappa),
            method="dogbox",
            bounds=([0, 0], [np.inf, np.inf]),
        )
        self.alpha = param[0]
        self.kappa = param[1]

        if self.alpha <= 0:
            self.alpha = 1
        if self.kappa <= 0:
            self.kappa = 1

    def calculate(self, ts, midprice):

        price = price
        # Descending order of price-timestamp quotes
        self._last_quotes = [
            {"timestamp": timestamp, "price": price}
        ] + self._last_quotes

        latest_processed_quote_idx = None
        # iterate over trades
        for trade in self._current_trade_sample:
            # iterate over quotes
            for i, quote in enumerate(self._last_quotes):
                # if quote happened before trade
                if quote["timestamp"] < trade.timestamp:
                    # if quote happened before latest processed quote
                    if (
                        latest_processed_quote_idx is None
                        or i < latest_processed_quote_idx
                    ):
                        latest_processed_quote_idx = i
                    trade = {
                        "price_level": abs(trade.price - float(quote["price"])),
                        "amount": trade.amount,
                    }

                    if quote["timestamp"] + 1 not in self._trade_samples.keys():
                        self._trade_samples[quote["timestamp"] + 1] = []

                    self._trade_samples[quote["timestamp"] + 1] += [trade]
                    break

        # THere are no trades left to process
        self._current_trade_sample = []
        # Store quotes that happened after the latest trade + one before
        if latest_processed_quote_idx is not None:
            self._last_quotes = self._last_quotes[0 : latest_processed_quote_idx + 1]

        if len(self._trade_samples.keys()) > self._sampling_length:
            timestamps = list(self._trade_samples.keys())
            timestamps.sort()
            timestamps = timestamps[-self._sampling_length :]

            trade_samples = {}
            for timestamp in timestamps:
                trade_samples[timestamp] = self._trade_samples[timestamp]
            self._trade_samples = trade_samples

        if self.is_sampling_buffer_full:
            self.c_estimate_intensity()
