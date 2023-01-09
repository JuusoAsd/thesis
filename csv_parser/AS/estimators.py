import logging
import numpy as np
from scipy.optimize import curve_fit


def curve_func(t, a, b):
    return a * np.exp(-b * t)


# you can run this by running python -m pytest tests/test_as.py -v -s
class IntensityEstimator:
    """
    lookback: how far back (in milliseconds) to look at trades, default: 1 day
    update_interval: how often to update the estimator (in milliseconds), default: 1 minute

    lookback is a rough estimate that acts as upper bound on how far back we look at trades,
    when we are past the lookback period, we switch to shorter set of trades and start estimating intenstity from that
    """

    def __init__(self, lookback=(1000 * 60 * 60 * 24), update_interval=(1000 * 60)):
        self.lookback = lookback
        self.update_interval = update_interval
        self.previous_update = 0
        self.lookback_start_time = 0
        self.trades = {}
        self.trades_short = {}
        self.kappa = 1
        self.alpha = 1
        self.trade_count = 0

    # we start by updating main estimator with trades and recording the value every record_interval
    # after lookback_half, we also start updating secondary estimator with the same trades
    # after lookback, we set the secondary as main and reset secondary
    def update_trades(self, new_trades):
        # new trades is an array containing [timestamp, trade price, trade amount and current mid price when trade took place]
        logging.debug(f"new trades: {new_trades}")
        for ts, trade_price, amount, mid_price in new_trades:
            if self.lookback_start_time == 0:
                self.lookback_start_time = ts
            price_diff = round(abs(trade_price - mid_price), 5)
            if price_diff != 0:
                self.trade_count += 1
                # record the trade on self.trades
                if price_diff in self.trades:
                    self.trades[price_diff] += amount
                else:
                    self.trades[price_diff] = amount
                # record the trade on shorter lookback
                if ts >= self.lookback_start_time + self.lookback / 2:
                    if price_diff in self.trades_short:
                        self.trades_short[price_diff] += amount
                    else:
                        self.trades_short[price_diff] = amount
                # set shorter lookback as main lookback
                if ts >= self.lookback_start_time + self.lookback:
                    self.trades = self.trades_short
                    self.trades_short = {}
                    self.lookback_start_time = ts
                    logging.info(
                        f"Switching estimators at {ts} with {self.trade_count} trades"
                    )
                    self.trade_count = 0

    def calculate_current_values(self):
        price_levels = np.array(list(self.trades.keys()))
        price_levels.sort()
        # reverse price_levels
        price_levels = price_levels[::-1]

        lambdas = []
        for i in price_levels:
            lambdas.append(self.trades[i])
        lambdas = np.array(lambdas)

        alpha, kappa = self.fit_curve(price_levels, lambdas)

        if alpha > 0 and kappa > 0:
            self.alpha = alpha
            self.kappa = kappa

    def fit_curve(self, price_levels, lambdas):
        try:
            param, _ = curve_fit(
                f=curve_func,
                xdata=price_levels,
                ydata=lambdas,
                p0=(self.alpha, self.kappa),
                method="dogbox",
                bounds=([0, 0], [np.inf, np.inf]),
            )
            alpha = param[0]
            kappa = param[1]
            return alpha, kappa

        except RuntimeError as e:
            logging.error(f"Failed estimating parameters for intensity")
        except Exception as e:
            logging.error(f"Error fitting curve for intensity")
            # logging.error(f"price_levels: {price_levels}")
            # logging.error(f"lambdas: {lambdas}")
            raise e

    def get_current_estimate(self, ts):
        if ts >= self.previous_update + self.update_interval:
            self.calculate_current_values()
            self.previous_update = ts
        return self.alpha, self.kappa


class VolatilityEstimator:
    """
    lookback: how far back (in milliseconds) to look at trades, default: 1 day
    return_aggregation: aggregation period for price returns (what is the time period between two price observations), default: 10 minutes
    update_interval: how often to update the estimator (in milliseconds), default: 1 minute
    """

    def __init__(
        self,
        lookback=(1000 * 60 * 60 * 24),
        return_aggregation=(1000 * 60 * 10),
        update_interval=(1000 * 60),
    ):
        self.lookback = lookback
        self.return_aggregation = return_aggregation
        self.update_interval = update_interval

        self.previous_ts = 0

        self.prices = []

        self.volatility = 0
        self.previous_update = 0

    def update_prices(self, new_price, ts):
        if self.previous_price == 0:
            self.previous_ts = ts
        elif ts >= self.previous_ts + self.return_aggregation:
            self.prices.append(new_price)
            self.previous_ts = ts

    def calculate_volatility_real(self):
        """
        Vol = std(returns)
        Annualized vol = vol * sqrt(aggregation periods per year)
            - aggregation period is in milliseconds
        """
        a = np.array(self.prices)
        return_arr = np.diff(a) / a[:, 1:]
        vol = np.std(return_arr)
        self.volatility = vol * np.sqrt(
            (1000 * 60 * 60 * 24 * 365) / self.return_aggregation
        )

    def calculate_volatility_hummingbot(self):
        """
        Calculates price volatility instead of percentage volatility
        """
        arr = np.array(self.prices)
        self.volatility = np.sqrt(np.sum(np.square(np.diff(arr))) / arr.size)

    def get_current_estimate(self, ts):
        if ts >= self.previous_update + self.update_interval:
            self.calculate_volatility_hummingbot()
            self.previous_update = ts
        return self.volatility


# OLD: comes from hummingbot
def calculate(self, ts, midprice):

    price = price
    # Descending order of price-timestamp quotes
    self._last_quotes = [{"timestamp": timestamp, "price": price}] + self._last_quotes

    latest_processed_quote_idx = None
    # iterate over trades
    for trade in self._current_trade_sample:
        # iterate over quotes
        for i, quote in enumerate(self._last_quotes):
            # if quote happened before trade
            if quote["timestamp"] < trade.timestamp:
                # if quote happened before latest processed quote
                if latest_processed_quote_idx is None or i < latest_processed_quote_idx:
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
