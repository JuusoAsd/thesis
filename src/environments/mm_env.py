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


class CurrentStateBaseClass:
    def __init__(self):
        self.timestamp = 1


class MMEnv(gym.Env):
    def __init__(
        self,
        state_folder,
        policy,
        capital=1000,
        step_interval=10_000,
        price_decimals=4,
        tick_size=0.0001,
        inventory_target=0,
        output_type=None,
        params={},
        policy_params={},
        logger=False,
        logging=False,
    ):
        self.policy = policy(self, **policy_params)
        self.state_folder = state_folder

        self.logger = logger
        self.logging = logging

        self.capital = capital
        self.step_interval = step_interval
        self.price_decimals = price_decimals
        self.output_type = output_type
        self.tick_size = tick_size
        self.inventory_target = inventory_target
        assert output_type is not None, "output_type must be specified"

        for k, v in params.items():
            setattr(self, k, v)

        # to observe the external environment we need best bid and ask and possible trades (price and size), for internal just inventory
        # Everything else depends on the inputs of the policy we are using
        # observations are read from raw data as is but then normalized to be in the given ranges
        required_observations = {
            "best_bid": [0, 1000],
            "best_ask": [0, 1000],
            "inventory": [-1, 1],
        }

        self.observation_space = self.policy.get_observation_space(
            required_observations
        )
        self.action_space = self.policy.get_action_space()

        self.reset()
        return None

    def reset(self):
        self.counter = 0
        self.error_counter = 0
        self.trade_counter = 0
        self.quote_asset = self.capital
        self.base_asset = 0
        self.previous_ts = 0

        self.bid = 0
        self.bid_size = 0
        self.ask = 0
        self.ask_size = 0

        self.state_manager = FileManager(self.state_folder, self.output_type)
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
        self.previous_state = self.current_state
        return self.policy.get_observation()

    def step(self, action):
        """
        - 1) Receive action in internal format
        - 2) Execute action
        - 3) Get the latest state of the environment
        - 4) Calculate reward on latest state
        - 5) Convert actions and observations to normalized format
        - 6) return observation, reward, and stuff
        """
        # 2) apply action how policy would apply it
        self.policy.apply_action(action)

        start_value = self.get_total_value()

        # 3)
        previous_state = self.current_state
        self.current_state = self.state_manager.get_next_event()
        if self.current_state is None:
            self.current_state = previous_state
            observation = self.policy.get_observation()
            return observation, 0, True, {}

        # 3) between steps, some time passes determined by step_interval
        while self.current_state.timestamp - self.previous_ts < self.step_interval:
            self.execute_market_orders()
            self.execute_limit_orders()
            self.previous_state = self.current_state
            self.current_state = self.state_manager.get_next_event()
            if self.current_state is None:
                self.current_state = previous_state
                observation = self.policy.get_observation()
                return observation, 0, True, {}
        self.previous_ts = self.current_state.timestamp

        # 4)
        reward = self.get_total_value() - start_value

        # 5)
        observation = self.policy.get_observation()
        if abs(self.get_inventory(self.current_state.mid_price)) > 1:
            logging.debug(f"Inventory too high, liquidated")
            return observation, -100, True, {}

        return observation, reward, False, {}

    def execute_market_orders(self):
        """
        market orders are executed at at best bid-ask
        """
        if self.ask <= self.current_state.best_bid and self.ask_size > 0:
            sell_price = self.current_state.best_bid
            sell_size = self.ask_size
            self._sell(sell_price, sell_size)
            if self.logging:
                self.logger.info(
                    f"action,{self.current_state.timestamp},market_sell,{sell_price},{sell_size},{self.current_state.mid_price},{self.base_asset},{self.quote_asset}"
                )

        if self.bid >= self.current_state.best_ask and self.bid_size > 0:
            buy_price = self.current_state.best_ask
            buy_size = self.bid_size
            self._buy(buy_price, buy_size)
            if self.logging:
                self.logger.info(
                    f"action,{self.current_state.timestamp},market_buy,{buy_price},{buy_size},{self.current_state.mid_price},{self.base_asset},{self.quote_asset}"
                )

    def execute_limit_orders(self):
        if self.current_state.trade_size != 0:
            if self.current_state.trade_price <= self.bid and self.bid_size > 0:
                bid_price = self.bid
                bid_size = self.bid_size
                self._buy(bid_price, bid_size)
                if self.logging:
                    self.logger.info(
                        f"action,{self.current_state.timestamp},limit_buy,{bid_price},{bid_size},{self.current_state.mid_price},{self.base_asset},{self.quote_asset}"
                    )

            elif self.current_state.trade_price >= self.ask and self.ask_size > 0:
                ask_price = self.ask
                ask_size = self.ask_size
                self._sell(ask_price, ask_size)
                if self.logging:
                    self.logger.info(
                        f"action,{self.current_state.timestamp},limit_sell,{ask_price},{ask_size},{self.current_state.mid_price},{self.base_asset},{self.quote_asset}"
                    )

    def get_total_value(self):
        value = self.quote_asset + self.base_asset * self.current_state.mid_price
        if self.quote_asset < 0 or self.base_asset < 0:
            return value
        else:
            return value

    def get_current_value(self):
        return (
            self.quote_asset
            + self.base_asset
            * (self.previous_state.best_bid + self.previous_state.best_ask)
            / 2
        )

    def _buy(self, price, amount):
        self.quote_asset -= price * amount
        self.base_asset += amount
        self.bid = 0
        self.bid_size = 0
        self.trade_counter += 1

    def _sell(self, price, amount):
        self.quote_asset += price * amount
        self.base_asset -= amount
        self.ask = 1_000_000_000
        self.ask_size = 0
        self.trade_counter += 1

    def get_values(self):
        return [
            self.quote_asset,
            self.base_asset,
            self.current_state.mid_price,
            self.bid,
            self.bid_size,
            self.ask,
            self.ask_size,
            self.exec_market_buy,
            self.exec_market_sell,
            self.exec_limit_buy,
            self.exec_limit_sell,
        ]

    def round_prices(self):
        self.bid = round(self.bid, self.price_decimals)
        self.ask = round(self.ask, self.price_decimals)

    def get_inventory(self, mid_price):
        return (
            self.base_asset / (self.base_asset + self.quote_asset / mid_price)
            - self.inventory_target
        )
