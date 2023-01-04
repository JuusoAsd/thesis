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
        state_folder,
        agent: AgentBaseClass,
        agent_parameters={},
        capital=1000,
        aggregation_window=10,
    ):
        self.agent = agent(**agent_parameters)
        self.state_folder = state_folder
        self.capital = capital
        self.aggregation_window = aggregation_window
        self.reset()

    def reset(self):
        self.counter = 0
        self.error_counter = 0
        self.quote_asset = self.capital
        self.base_asset = 0
        self.previous_ts = 0

        self.bid = 0
        self.bid_size = 0
        self.ask = 0
        self.ask_size = 0

        self.state_manager = FileManager(self.state_folder, self.agent.data_type)
        """
        Current state constains all the data needed to execute trades
          - timestamp
          - best bid (for execution of market sell orders)
          - best ask (for execution of market buy orders)
          - trade price (0 if no trade)
          - trade size (0 if no trade)
        It also contains agent-specific data
          """
        self.current_state = self.state_manager.get_next_event()

    def step(self, action):
        """
        Agent has set its desired b/a in the previous state
        Multiple events happen in environment based on latest data in following order:
            - market orders by agent are executed
            - best bid and ask, midprice are updated
            - possible trades are executed againts agent's limit orders
        Agent sets new desired b/a based on the latest state

        Step has some configuration associated with it:
        - aggregation_window (10): how many milliseconds minimum between agent actions
        """
        self.counter += 1
        if self.counter % 1_000_000 == 0:
            print(f"Counter: {self.counter}, Error counter: {self.error_counter}")
        if self.previous_ts == 0:
            self.previous_ts = self.current_state.timestamp

        # Only update orders when aggregation_window has passed since previous update, still execute trades if exist
        while self.current_state.timestamp - self.previous_ts < self.aggregation_window:
            self.execute_market_orders()
            self.execute_limit_orders()
            self.current_state = self.state_manager.get_next_event()
            if self.current_state is None:
                return True
        self.previous_ts = self.current_state.timestamp
        self.agent.step()

    def execute_market_orders(self):
        """
        market orders are executed at at best bid-ask
        """
        if self.ask <= self.current_state.best_bid and self.ask_size > 0:
            self._sell(self.current_state.best_bid, self.ask_size)

        if self.bid >= self.current_state.best_ask and self.bid_size > 0:
            self._buy(self.current_state.best_ask, self.bid_size)

    def execute_limit_orders(self):
        if self.current_state.trade is not None:
            if self.current_state.trade.price <= self.bid:
                self.buy(self.bid, self.bid_size)
            elif self.current_state.trade.price >= self.ask:
                self.sell(self.ask, self.ask_size)

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
        self.ask = np.inf
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
