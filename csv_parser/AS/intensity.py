import numpy as np
from scipy.optimize import curve_fit


def curve_func(a, b, t):
    return a * np.exp(-b * t)


# you can run this by running python -m pytest tests/test_as.py
class IntensityEstimator:
    def __init__(self):
        self.trades = {}
        self.kappa = 2
        self.alpha = 2

    def estimate_intensity(self, new_trades):
        for trade, amount in new_trades:
            if trade in self.trades:
                self.trades[trade] += amount
            else:
                self.trades[trade] = amount

        price_levels = np.array(list(self.trades.keys()))
        price_levels.sort()
        price_levels = price_levels[:: -len(price_levels)]

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

        if self.alpha < 0:
            self.alpha = 1
        if self.kappa < 0:
            self.kappa = 1
