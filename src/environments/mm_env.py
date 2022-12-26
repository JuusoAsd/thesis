# MM_env is base environment that has knowledge of order book and executed trades
# It always contains an agent that has its own strategy of setting orders
# Based on the orders set by the agent and the current datapoints, the environment executes trades
# TODO: this should be done as vectorized environment but easier to handle calculations this way first

import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import time
import pandas as pd
import numpy as np
import gym
from gym import spaces
from gym.wrappers.normalize import NormalizeObservation
import csv
import sys

from src.environments.util import Trade, FileManager


class AgentBaseClass:
    def __init__(self, env):
        # all agents have access to the environment
        self.env = env
        self.reset()

    def reset(self):
        raise NotImplementedError(
            f"reset not implemented for {self.__class__.__name__}"
        )

    def step(self):
        raise NotImplementedError(f"step not implemented for {self.__class__.__name__}")


class MMEnv(gym.Env):
    def __init__(
        self,
        trades_folder,
        orderbook_folder,
        agent: AgentBaseClass,
        agent_parameters={},
        capital=1000,
    ):
        self.agent = agent(**agent_parameters)
        self.trades_folder = trades_folder
        self.orderbook_folder = orderbook_folder
        self.capital = capital
        self.reset()

    def reset(self):
        self.counter = 0
        self.error_counter = 0
        self.quote_asset = self.capital

        self.bid = 0
        self.bid_size = 0
        self.ask = 0
        self.ask_size = 0

        # self.exec_market_buy = 0
        # self.exec_market_sell = 0
        # self.exec_limit_buy = 0
        # self.exec_limit_sell = 0

        self.trades_manager = FileManager(self.trades_folder, Trade)
        self.orderbook_manager = FileManager(self.orderbook_folder)

        self.current_trade = self.trades_manager.get_next_event()

    def step(self, action):
        """
        Agent set its desired b/a in the previous state
        Multiple events happen in environment based on latest data in following order:
            - market orders by agent are executed
            - best bid and ask, midprice are updated
            - possible trades are executed againts agent's limit orders
        Agent sets new desired b/a based on the latest state
        """

        """
        Step has some configuration associated:
        - time_aggregation (10): how many milliseconds between each time agent can act, this is to make lag more realistic 
        """
        self.counter += 1
        if self.counter % 1_000_000 == 0:
            print(f"Counter: {self.counter}, Error counter: {self.error_counter}")

        self.execute_market_orders()
        self.execute_limit_orders()

    def execute_market_orders(self):
        """
        market orders are executed at at best b/a
        """
        if self.ask < self.mid_price and self.ask_size > 0:
            # self._sell(self.mid_price, self.ask_size)
            self._sell(self.ask, self.ask_size)
            self.exec_market_sell = 1
        if self.bid > self.mid_price and self.bid_size > 0:
            # self._buy(self.mid_price, self.bid_size)
            self._buy(self.bid, self.bid_size)
            self.exec_market_buy = 1

    def execute_limit_orders(self, timestamp):
        while self.current_trade.timestamp <= timestamp:
            self.execute_trade(self.current_trade)
            self.current_trade = self.trades_manager.get_next_event()

    def execute_trade(self, trade):
        # check if our current bid is hit
        if self.bid >= trade.price and self.bid_size > 0:
            self._buy(trade.price, self.bid_size)
            self.exec_limit_buy = 1
        elif self.ask <= trade.price and self.ask_size > 0:
            self._sell(trade.price, self.ask_size)
            self.exec_limit_sell = 1

    def get_total_value(self, mid_price):
        value = self.quote_asset + self.base_asset * mid_price
        if self.quote_asset < 0 or self.base_asset < 0:
            return value, True
        else:
            return value, False

    def _buy(self, price, amount):
        self.quote_asset -= price * amount
        self.base_asset += amount
        self.bid = 0
        self.bid_size = 0

    def _sell(self, price, amount):
        self.quote_asset += price * amount
        self.base_asset -= amount
        self.ask = 0
        self.ask_size = 0

    def get_values(self):
        return [
            self.quote_asset,
            self.base_asset,
            self.mid_price,
            self.bid,
            self.bid_size,
            self.ask,
            self.ask_size,
            self.exec_market_buy,
            self.exec_market_sell,
            self.exec_limit_buy,
            self.exec_limit_sell,
        ]
