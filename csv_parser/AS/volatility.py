import numpy as np


class VolatilityEstimator:
    def __init__(self):
        self.prices = []
        self.volatility = 0
        self.prices_count = 0

    def update_prices(self, new_price):
        # new_price is a variable containing one new price
        self.prices.append(new_price)
        self.prices_count += 1

    def count_volatility(self):
        # TO DO: Deciding sample size.. Similar to intensity calculations probably?
        price_sample = np.array(self.prices)
        vol = np.sqrt(np.sum(np.square(np.diff(price_sample)))) / price_sample.size
        self.volatility = vol
        self.prices = []
        self.prices_count = 0
