import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import time
import pandas as pd
import numpy as np
import gym
from gym import spaces
from gym.wrappers.normalize import NormalizeObservation
import csv
import sys


class Trade:
    # 557111802,1.2373,10.0,12.373,1640044800076,false
    def __init__(self, input_list):
        self.timestamp = int(input_list[4])
        self.price = float(input_list[1])
        self.size = float(input_list[2])
        # self.buyer_maker = input_list[4] == "true"


class FileManager:
    def __init__(self, folder_path, output_type=None):
        self.path = folder_path
        self.files = os.listdir(self.path)
        self.files.sort()
        self.create_output = output_type

    def get_next_file(self):
        if len(self.files) > 0:
            self.iterator = iter(open(os.path.join(self.path, self.files.pop(0))))
            self.iterator.__next__()
        else:
            return None

    def get_next_event(self, start=False):
        try:
            val = next(self.iterator)
        except:
            self.get_next_file()
            val = next(self.iterator)
        if self.create_output is None:
            return val.rstrip().split(",")
        else:
            return self.create_output(val.rstrip().split(","))


class MMFullOrderbookSnapshotEnv(gym.Env):

    def __init__(self, trades_folder, orderbook_folder, mid_price, capital=1000):
        """
        Goal of this environment is to do deep reinforcement learning on the full orderbook and possibly in the future some infered statistics of it to help converging.

        PROBLEMS & DECISIONS:
            - SPARSE DATA: orderbook data is very high frequency, and therefore quite information poor. However we do not know what is important, and therefore let model decide.
            - LATENCY: Only allow model to act on every nth step. This is to make it somewhat comparable to real-world, where updates and reactions are not instant.
        """
        self.average_price = mid_price
        self.trades_folder = trades_folder
        self.orderbook_folder = orderbook_folder
        self.capital = capital
        self.reset()
        '''
        Normalize action spaces as most reinforcement learning algorithms rely on a Gaussian distribution
        for continuous actions. Best to have an interval range of 2 (low = -1, high = 1).
        However, normalizing the action space did not provide any better results so far..
        '''
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(102,), dtype=np.float32
        )

    def reset(self):
        self.counter = 0
        self.error_counter = 0
        self.quote_asset = self.capital
        self.base_asset = self.quote_asset / self.average_price
        self.previous_value = self.get_total_value(self.average_price)[0]
        self.mid_price = 0

        self.bid = 0
        self.bid_size = 0
        self.ask = 0
        self.ask_size = 0
        self.exec_market_buy = 0
        self.exec_market_sell = 0
        self.exec_limit_buy = 0
        self.exec_limit_sell = 0
        self.trade_asks = []
        self.trade_bids = []
        self.trades_manager = FileManager(self.trades_folder, Trade)
        self.orderbook_manager = FileManager(self.orderbook_folder)

        self.current_trade = self.trades_manager.get_next_event()

        return np.array(
            np.random.uniform(low=-1, high=1, size=(102,)), dtype=np.float32
        )

    def step(self, action):
        self.eof = 0
        self.exec_market_buy = 0
        self.exec_market_sell = 0
        self.exec_limit_buy = 0
        self.exec_limit_sell = 0
        self.counter += 1
        if self.counter % 1_000_000 == 0:
            print(f"Counter: {self.counter}, Error counter: {self.error_counter}")
        self.bid = action[0]
        self.bid_size = action[1]
        self.ask = action[2]
        self.ask_size = action[3]
        try:
            state = self.orderbook_manager.get_next_event()
        except StopIteration:
            res = self.reset()
            self.eof = 1
        timestamp = int(state[0])
        self.mid_price = float(state[1])
        orderbook = list(state[2:])

        self.execute_market_orders()
        self.execute_limit_orders(timestamp)
        self.calculate_OSI()

        current_value, liquidated = self.get_total_value(self.mid_price)
        if liquidated:
            reward = -100
            self.quote_asset = 0
            self.base_asset = 0
            done = True
        else:
            reward = current_value - self.previous_value
            done = False

        current_state = np.array(
            [self.quote_asset, self.base_asset] + orderbook
        ).astype(np.float32)
        if self.eof == 1:
            return res, 0, True, {}
        return current_state, reward, done, {}

    def execute_market_orders(self):
        if self.ask < self.mid_price and self.ask_size > 0:
            # self._sell(self.mid_price, self.ask_size)
            self._sell(self.ask, self.ask_size)
            self.exec_market_sell = 1
        if self.bid > self.mid_price and self.bid_size > 0:
            # self._buy(self.mid_price, self.bid_size)
            self._buy(self.bid, self.bid_size)
            self.exec_market_buy = 1

    def execute_limit_orders(self, timestamp):
        # Does this assume that our agent participates in every single trade?
        while self.current_trade.timestamp <= timestamp:
            self.execute_trade(self.current_trade)

            # adding these lines of code to append the sizes of bid and ask trades for OSI calculation
            if self.current_trade.price < self.mid_price:
                self.trade_bids.append([self.current_trade.timestamp, self.current_trade.size])
            elif self.current_trade.price > self.mid_price:
                self.trade_asks.append([self.current_trade.timestamp, self.current_trade.size])

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


    def calculate_OSI(self):
        '''
        Function for calculating OSI. Currently updates OSI every hour.
        self.trade_bids/asks are two dimensional arrays formed like this [[timestamp, size]]
        This function sorts the trades by size and takes the 90% quantile sized trades for OSI calculation.
        '''
        update_frequency = 60 * 60
        if ((self.trade_bids[0][0] - self.trade_bids[-1][0])) >= update_frequency or ((self.trade_asks[0][0] - self.trade_asks[-1][0])) >= update_frequency:
            bids, asks = np.array(self.trade_bids), np.array(self.trade_asks)
            bids, asks = bids[:,1], asks[:,1]
            bids_sorted_index, asks_sorted_index  = np.argsort(bids), np.argsort(asks)
            sorted_bids, sorted_asks  = bids[bids_sorted_index], asks[asks_sorted_index]
            bids_values, asks_values = round(bids.size / 10), round(asks.size / 10)
            decile90_bids, decile90_asks = np.sum(sorted_bids[-bids_values : ]), np.sum(sorted_asks[-asks_values : ])
            OSI = 100 * ((decile90_bids - decile90_asks) / (decile90_bids + decile90_asks))
            self.trade_bids = []
            self.trade_asks = []


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


#trades = "/home/juuso/Documents/gradu/parsed_data/trades"
trades = r"C:\Users\Ville\Documents\gradu\data\trades"
#orderbook = r"/home/juuso/Documents/gradu/parsed_data/orderbook"
orderbook = r"C:\Users\Ville\Documents\gradu\parsed_data\orderbook"
#model_path = "/home/juuso/Documents/gradu/src/models/ppo_mm_full_orderbook"
model_path = r"C:\Users\Ville\Documents\gradu\gradu\src\models"
#normalize_path = ("/home/juuso/Documents/gradu/src/models/mm_full_orderbook_normalize.pkl")
normalize_path = r"C:\Users\Ville\Documents\gradu\gradu\src\models\mm_full_orderbook_normalize.pkl"

'''for vectorized envs the following could work (it did not):
# num_cpu = 4
# env = SubprocVecEnv([MMFullOrderbookSnapshotEnv(trades, orderbook, 1.2) for i in range(num_cpu)])
'''
env = MMFullOrderbookSnapshotEnv(trades, orderbook, 1.2)
env = NormalizeObservation(env)
print(env.reset())
# check_env(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2_000)

# Creating a separate environment for evaluation
eval_env = MMFullOrderbookSnapshotEnv(trades, orderbook, 1.2)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')


def train():
    model.learn(total_timesteps=2_000)
    model.save(model_path)

    # with normalized env, also need to save the normalization parameters
    env.save(normalize_path)


# def test():
#     model.load("/home/juuso/Documents/gradu/src/models/ppo_mm_full_orderbook")
#     obs = env.reset()
#     result_path = (
#         "/home/juuso/Documents/gradu/parsed_data/results/ppo_mm_full_orderbook.csv"
#     )
#     env = VecNormalize.load(stats_path, env)
#     #  do not update them at test time
#     env.training = False
#     # reward normalization is not needed at test time
#     env.norm_reward = False

#     with open(result_path, "w") as f:
#         writer = csv.writer(f)
#         while True:
#             action, _states = model.predict(obs)
#             obs, rewards, dones, info = env.step(action)
#             values = env.get_values()
#             writer.writerow(values)
#             if dones:
#                 break


if __name__ == "__main__":
    globals()[sys.argv[1]]()
