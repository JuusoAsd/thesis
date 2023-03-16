import logging
import numpy as np
from scipy.optimize import curve_fit
import queue
import pandas as pd


def curve_func(t, a, b):
    return a * np.exp(-b * t)


class EstimatorABC:
    def __init__(self, lookback, update_interval):
        self.lookback = lookback
        self.update_interval = update_interval

    def update(self, **args):
        raise NotImplementedError("update_values not implemented")

    def calculate_values(self, **args):
        raise NotImplementedError("calculate_values not implemented")

    def get_value(self, **args):
        raise NotImplementedError("get_value not implemented")


# you can run this by running python -m pytest tests/test_as.py -v -s
class IntensityEstimator(EstimatorABC):
    """
    lookback: how far back (in milliseconds) to look at trades, default: 1 day
    update_interval: how often to update the estimator (in milliseconds), default: 1 minute
    """

    def __init__(self, lookback=(1000 * 60 * 60 * 24), update_interval=(1000 * 60)):
        super().__init__(lookback, update_interval)
        self.previous_update = 0
        self.trades = []
        self.trades
        self.kappa = 1
        self.alpha = 1
        self.trade_count = 0

    def update(self, new_trades):
        # new trades is an array containing [timestamp, trade price, trade amount and current mid price when trade took place]
        # trades are kept in a queue, so the oldest trade is removed when it is older than lookback
        # the self.trades dictionary is used to calculate the intensity, when oldest trade is popped it is also removed
        logging.debug(f"new trades: {new_trades}")
        for ts, trade_price, amount, mid_price in new_trades:
            price_diff = round(abs(trade_price - mid_price), 5)
            if price_diff != 0:
                self.trades.append([ts, price_diff, amount])

    def calculate_values(self):
        trades = (
            pd.DataFrame(self.trades, columns=["ts", "price_diff", "amount"])
            .groupby("price_diff")
            .sum()
        )
        price_levels = np.array(trades.index)[::-1]
        lambdas = np.array(trades["amount"])[::-1]

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
            return self.alpha, self.kappa
        except Exception as e:
            logging.error(f"Error fitting curve for intensity")
            raise e

    def get_value(self, ts):
        if ts >= self.previous_update + self.update_interval:
            # prune
            prices = np.array(self.trades)
            self.trades = prices[prices[:, 0] >= ts - self.lookback].tolist()
            self.calculate_values()
            self.previous_update = ts
        return self.alpha, self.kappa


class VolatilityEstimator(EstimatorABC):
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
        super().__init__(lookback, update_interval)
        self.return_aggregation = return_aggregation

        self.previous_ts = 0

        # self.prices = []
        self.prices = []

        self.volatility = 0
        self.previous_update = 0

    def update(self, new_price, ts):
        if ts >= self.previous_ts + self.return_aggregation:
            self.prices.append([ts, new_price])
            self.previous_ts = ts

    def calculate_volatility_real(self):
        """
        Vol = std(returns)
        Annualized vol = vol * sqrt(aggregation periods per year)
            - aggregation period is in milliseconds
        """
        a = np.array(self.price_queue.queue)
        return_arr = np.diff(a) / a[:, 1:]
        vol = np.std(return_arr)
        self.volatility = vol * np.sqrt(
            (1000 * 60 * 60 * 24 * 365) / self.return_aggregation
        )

    def calculate_values(self):
        """
        Calculates price volatility instead of percentage volatility
        """
        arr = np.array(self.prices)[:, 1]
        self.volatility = np.sqrt(np.sum(np.square(np.diff(arr))) / arr.size)

    def get_value(self, ts):
        if ts >= self.previous_update + self.update_interval:
            # prune old prices
            prices = np.array(self.prices)
            self.prices = prices[prices[:, 0] >= ts - self.lookback].tolist()
            self.calculate_values()
            self.previous_update = ts
        return self.volatility


class OSIEstimator(EstimatorABC):
    def __init__(
        self,
        lookback=1000 * 60 * 60 * 24,
        update_interval=1000 * 60 * 60,
    ):
        super().__init__(lookback, update_interval)
        self.buys = []
        self.sells = []
        self.previous_update = 0

    def update(self, trades):
        # trade is list of lists [[timestamp, price, amount, mid_price]]
        for trade in trades:
            ts, price, amount, mid_price = trade

            # sides are determined by price relative to mid_price, based on what side take is on
            if price < mid_price:
                self.sells.append([ts, amount])
            elif price > mid_price:
                self.buys.append([ts, amount])

    def calculate_values(self):
        """
        Function for calculating OSI. Currently updates OSI every hour.
        self.trade_bids/asks are two dimensional arrays formed like this [[timestamp, size]]
        This function sorts the trades by size and takes the 90% quantile sized trades for OSI calculation.
        """
        buy_qty, sell_qty = np.array(self.buys)[:, 1], np.array(self.sells)[:, 1]
        decile_buys = buy_qty[(buy_qty > np.percentile(buy_qty, 90))].sum()
        decile_sells = sell_qty[(sell_qty < np.percentile(sell_qty, 90))].sum()

        osi = 100 * ((decile_buys - decile_sells) / (decile_buys + decile_sells))
        self.osi = osi

    def get_value(self, ts):
        if ts >= self.previous_update + self.update_interval:
            # prune trades that are too old
            buy_arr = np.array(self.buys)
            self.buys = buy_arr[buy_arr[:, 0] > (ts - self.lookback)].tolist()

            sell_arr = np.array(self.sells)
            self.sells = sell_arr[sell_arr[:, 0] > (ts - self.lookback)].tolist()

            self.calculate_values()
            self.previous_update = ts
        return self.osi


# # you can run this by running python -m pytest tests/test_as.py -v -s
# class IntensityEstimator(EstimatorABC):
#     """
#     lookback: how far back (in milliseconds) to look at trades, default: 1 day
#     update_interval: how often to update the estimator (in milliseconds), default: 1 minute
#     """

#     def __init__(self, lookback=(1000 * 60 * 60 * 24), update_interval=(1000 * 60)):
#         super().__init__(lookback, update_interval)
#         self.previous_update = 0
#         self.trades = {}
#         self.trade_queue = queue.Queue()
#         self.kappa = 1
#         self.alpha = 1
#         self.trade_count = 0

#     def update(self, new_trades):
#         # new trades is an array containing [timestamp, trade price, trade amount and current mid price when trade took place]
#         # trades are kept in a queue, so the oldest trade is removed when it is older than lookback
#         # the self.trades dictionary is used to calculate the intensity, when oldest trade is popped it is also removed
#         logging.debug(f"new trades: {new_trades}")
#         for ts, trade_price, amount, mid_price in new_trades:
#             price_diff = round(abs(trade_price - mid_price), 5)
#             self.trade_queue.put((ts, price_diff, amount))
#             if price_diff != 0:
#                 self.trade_count += 1
#                 # record the trade on self.trades
#                 if price_diff in self.trades:
#                     self.trades[price_diff] += amount
#                 else:
#                     self.trades[price_diff] = amount

#                 if ts >= self.trade_queue.queue[0][0] + self.lookback:
#                     oldest = self.trade_queue.get()
#                     self.trades[oldest[1]] -= oldest[2]
#                     if self.trades[oldest[1]] == 0:
#                         self.trades.pop(oldest[1])
#                     elif self.trades[oldest[1]] < 0:
#                         raise ValueError("trade amount is negative")

#     def calculate_values(self):
#         price_levels = np.array(list(self.trades.keys()))
#         price_levels.sort()
#         # reverse price_levels
#         price_levels = price_levels[::-1]

#         lambdas = []
#         for i in price_levels:
#             lambdas.append(self.trades[i])
#         lambdas = np.array(lambdas)

#         alpha, kappa = self.fit_curve(price_levels, lambdas)

#         if alpha > 0 and kappa > 0:
#             self.alpha = alpha
#             self.kappa = kappa

#     def fit_curve(self, price_levels, lambdas):
#         try:
#             param, _ = curve_fit(
#                 f=curve_func,
#                 xdata=price_levels,
#                 ydata=lambdas,
#                 p0=(self.alpha, self.kappa),
#                 method="dogbox",
#                 bounds=([0, 0], [np.inf, np.inf]),
#             )
#             alpha = param[0]
#             kappa = param[1]
#             return alpha, kappa

#         except RuntimeError as e:
#             logging.error(f"Failed estimating parameters for intensity")
#             return self.alpha, self.kappa
#         except Exception as e:
#             logging.error(f"Error fitting curve for intensity")
#             # logging.error(f"price_levels: {price_levels}")
#             # logging.error(f"lambdas: {lambdas}")
#             raise e

#     def get_value(self, ts):
#         if ts >= self.previous_update + self.update_interval:
#             self.calculate_values()
#             self.previous_update = ts
#         return self.alpha, self.kappa


# class VolatilityEstimator(EstimatorABC):
#     """
#     lookback: how far back (in milliseconds) to look at trades, default: 1 day
#     return_aggregation: aggregation period for price returns (what is the time period between two price observations), default: 10 minutes
#     update_interval: how often to update the estimator (in milliseconds), default: 1 minute
#     """

#     def __init__(
#         self,
#         lookback=(1000 * 60 * 60 * 24),
#         return_aggregation=(1000 * 60 * 10),
#         update_interval=(1000 * 60),
#     ):
#         super().__init__(lookback, update_interval)
#         self.return_aggregation = return_aggregation

#         self.previous_ts = 0

#         # self.prices = []
#         self.price_queue = queue.Queue()

#         self.volatility = 0
#         self.previous_update = 0

#     def update(self, new_price, ts):
#         if ts >= self.previous_ts + self.return_aggregation:
#             # self.prices.append(new_price)
#             self.price_queue.put(np.array([ts, new_price]))
#             # self.ts_queue.put(ts)
#             self.previous_ts = ts

#         if not self.price_queue.empty():
#             if ts > self.price_queue.queue[0][1] + self.lookback:
#                 self.price_queue.get()
#             # self.ts_queue.get()

#     def calculate_volatility_real(self):
#         """
#         Vol = std(returns)
#         Annualized vol = vol * sqrt(aggregation periods per year)
#             - aggregation period is in milliseconds
#         """
#         a = np.array(self.price_queue.queue)
#         return_arr = np.diff(a) / a[:, 1:]
#         vol = np.std(return_arr)
#         self.volatility = vol * np.sqrt(
#             (1000 * 60 * 60 * 24 * 365) / self.return_aggregation
#         )

#     def calculate_values(self):
#         """
#         Calculates price volatility instead of percentage volatility
#         """
#         arr = np.array(self.price_queue.queue)
#         self.volatility = np.sqrt(np.sum(np.square(np.diff(arr))) / arr.size)

#     def get_value(self, ts):
#         if ts >= self.previous_update + self.update_interval:
#             self.calculate_values()
#             self.previous_update = ts
#         return self.volatility


# class OSIEstimator(EstimatorABC):
#     def __init__(
#         self,
#         lookback=1000 * 60 * 60 * 24,
#         update_interval=1000 * 60 * 60,
#     ):
#         super().__init__(lookback, update_interval)
#         self.buy_trades = queue.Queue()
#         self.sell_trades = queue.Queue()
#         self.previous_update = 0

#     def update(self, trades):
#         # trade is list of lists [[timestamp, price, amount, mid_price]]
#         for trade in trades:
#             ts, price, amount, mid_price = trade

#             # sides are determined by price relative to mid_price, based on what side take is on
#             if price < mid_price:
#                 self.sell_trades.put(np.array([ts, amount]))
#             elif price > mid_price:
#                 self.buy_trades.put(np.array([ts, amount]))

#             # remove trades that are too old
#             if not self.buy_trades.empty():
#                 if ts > self.buy_trades.queue[0][0] + self.lookback:
#                     self.buy_trades.get()
#             if not self.sell_trades.empty():
#                 if ts > self.sell_trades.queue[0][0] + self.lookback:
#                     self.sell_trades.get()

#     def calculate_values(self):
#         """
#         Function for calculating OSI. Currently updates OSI every hour.
#         self.trade_bids/asks are two dimensional arrays formed like this [[timestamp, size]]
#         This function sorts the trades by size and takes the 90% quantile sized trades for OSI calculation.
#         """

#         buy, sell = np.array(self.buy_trades.queue), np.array(self.sell_trades.queue)
#         buy_qty, sell_qty = buy[:, 1], sell[:, 1]
#         decile_buys = buy_qty[: (buy_qty < np.percentile(buy_qty, 90)).argmin()].sum()
#         decile_sells = sell_qty[
#             : (sell_qty < np.percentile(sell_qty, 90)).argmin()
#         ].sum()

#         osi = 100 * ((decile_buys - decile_sells) / (decile_buys + decile_sells))
#         self.osi = osi

#     def get_value(self, ts):
#         if ts >= self.previous_update + self.update_interval:
#             self.calculate_values()
#             self.previous_update = ts
#         return self.osi
