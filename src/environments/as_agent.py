import logging
import numpy as np
from src.environments.mm_env import AgentBaseClass, CurrentStateBaseClass
from src.environments.util import Trade


class ASData(CurrentStateBaseClass):
    def __init__(self, val):
        self.timestamp = int(val[0])
        self.best_bid = float(val[1])
        self.best_ask = float(val[2])

        trade_size = float(val[4])
        if trade_size > 0:
            self.trade = Trade(float(val[3]), trade_size)
        else:
            self.trade = None
        self.vol = float(val[5])
        self.intensity = float(val[6])
        self.mid_price = (self.best_ask + self.best_bid) / 2


class ASAgent(AgentBaseClass):
    """
    Agent that behaves based on Avellaneda-Stoikov model
    Requires trades and order book midprice and uses those to calculate the indifference price and spread
    """

    def __init__(self, env, risk_aversion):
        super().__init__(env)
        self.data_type = ASData
        self.risk_aversion = risk_aversion

    def reset(self):
        pass

    def get_inventory(self, mid_price):
        return self.env.base_asset + self.env.base_asset / mid_price

    def step(self):
        """
        Set bid and ask prices based on the current state

        reservation_price = mid_price - inventory * vol ^ 2 * risk_aversion (* terminal time)
        spread = vol ^ 2 * risk_aversion (* terminal time) + 2/risk_aversion * ln(1 + risk_aversion / intensity)
        """

        mid_price = round(self.env.current_state.mid_price, self.env.price_decimals)
        vol = self.env.current_state.vol
        intensity = self.env.current_state.intensity
        inventory = self.get_inventory(mid_price)

        reservation_price = mid_price - inventory * vol * self.risk_aversion
        spread = vol * self.risk_aversion + 2 / self.risk_aversion * np.log(
            1 + self.risk_aversion / intensity
        )

        self.env.bid = reservation_price - spread / 2
        self.env.ask = reservation_price + spread / 2
        self.env.round_prices()
        self.env.bid_size = 1
        self.env.ask_size = 1

        if self.env.logging:
            self.env.logger.info(
                f"update,{self.env.current_state.timestamp},{self.env.bid},{self.env.ask},{mid_price},{vol},{intensity},{inventory},{reservation_price},{spread}"
            )
        logging.debug(
            f"mid price: {mid_price}, bid: {self.env.bid}, ask: {self.env.ask}, inventory: {inventory}, vol: {vol}, intensity: {intensity}"
        )

    def small_step(self):
        # not used because intensity and volatility are pre-calculated to the data
        pass
