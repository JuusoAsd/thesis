import logging
import numpy as np
from scipy.optimize import curve_fit
import queue
import pandas as pd


def curve_func(t, a, b):
    return a * np.exp(-b * t)


def line_func(t, a, b):
    return -a * t + b


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
        """
        This produces different values than what would be result from estimating parameters from linear function
        """
        try:
            param, diag = curve_fit(
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

        if self.buys and self.sells == []:
            self.osi = 0

        elif self.buys == []:
            self.osi = -100

        elif self.sells == []:
            self.osi = 100

        else:
            try:
                buy_qty, sell_qty = (
                    np.array(self.buys)[:, 1],
                    np.array(self.sells)[:, 1],
                )
            except Exception as e:
                print(self.buys)
                print(self.sells)
                raise e
            decile_buys = buy_qty[(buy_qty > np.percentile(buy_qty, 90))].sum()
            decile_sells = sell_qty[(sell_qty < np.percentile(sell_qty, 90))].sum()

            osi = 100 * ((decile_buys - decile_sells) / (decile_buys + decile_sells))
            self.osi = osi

    def get_value(self, ts):
        if ts >= self.previous_update + self.update_interval:
            # prune trades that are too old
            if self.buys != []:
                buy_arr = np.array(self.buys)
                self.buys = buy_arr[buy_arr[:, 0] > (ts - self.lookback)].tolist()

            if self.sells != []:
                sell_arr = np.array(self.sells)
                self.sells = sell_arr[sell_arr[:, 0] > (ts - self.lookback)].tolist()

            self.calculate_values()
            self.previous_update = ts
        return round(self.osi, 2)


def _compare(price, size):
    est = IntensityEstimator()
    x = price
    y = np.log(size)

    covar = np.cov(x, y, bias=True)
    var = np.var(x)

    intercept = covar[0, 1] / var
    slope = np.mean(y) - intercept * np.mean(x)

    a_calc = np.exp(slope)
    k_calc = -intercept

    print(f"alpha: {a_calc}, kappa: {k_calc}")

    est.alpha = a_calc
    est.kappa = k_calc

    a, k = est.fit_curve(price, size)
    print(f"alpha: {a}, kappa: {k}")

    # calculate the residual sum of squares for both models
    rss1 = np.sum((y - (a_calc * np.exp(k_calc * x))) ** 2)
    rss2 = np.sum((y - (a * np.exp(k * x))) ** 2)

    print(f"rss calc: {rss1}, rss est: {rss2}, calculated is better: {rss1 < rss2}")


# def test_original():
#     """
#     Seems that using cov and var gives better fitting estimations than using curve fit
#     """
#     data = pd.read_csv(
#         "/Users/juusoahlroos/Documents/own/gradu/csv_parser/test_data.csv"
#     )
#     price_levels = data["price"].values
#     size = data["size"].values
#     _compare(price_levels, size)


# def test_no_tail(last_rows):
#     """
#     Curve fit and cov/var give similar results when using fewer rows
#     more rows seem to give better results for cov/var
#     Curve fit weights the first obs more heavily?
#     """
#     data = pd.read_csv(
#         "/Users/juusoahlroos/Documents/own/gradu/csv_parser/test_data.csv"
#     )
#     # only use last 100 rows
#     data = data.tail(last_rows)

#     price_levels = data["price"].values
#     size = data["size"].values
#     _compare(price_levels, size)


# def test_reverse_data():
#     """
#     Reversing data has no effect on the results
#     """
#     data = pd.read_csv(
#         "/Users/juusoahlroos/Documents/own/gradu/csv_parser/test_data.csv"
#     )
#     data = data.iloc[::-1]
#     price_levels = data["price"].values
#     size = data["size"].values
#     _compare(price_levels, size)


# def test_duplicate_obs(dup_count, dup_rows):
#     """
#     Duplicating first rows affects the results but not significantly
#     """
#     data = pd.read_csv(
#         "/Users/juusoahlroos/Documents/own/gradu/csv_parser/test_data.csv"
#     )
#     data = data.iloc[::-1]

#     # duplicate first 10 obs and add to data
#     for i in range(dup_count):
#         data = pd.concat([data.head(dup_rows), data], ignore_index=True)
#     price_levels = data["price"].values
#     size = data["size"].values
#     _compare(price_levels, size)


# if __name__ == "__main__":
#     # total_rows = len(
#     #     pd.read_csv("/Users/juusoahlroos/Documents/own/gradu/csv_parser/test_data.csv")
#     # )
#     # for i in range(10, 110, 10):
#     #     print(f"last {i} rows")
#     #     test_no_tail(i)
#     #     print("")
#     # test_no_tail(total_rows)
#     # test_reverse_data()
#     # test_duplicate_obs(200, 1)

#     # just using the first 10 obs seems to be producing the closest results
#     test_no_tail(10)
