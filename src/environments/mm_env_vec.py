import os
import logging
from itertools import product
import logging
import pandas as pd
import numpy as np
import gym
from src.environments.env_configs.rewards import BaseRewardClass
import src.environments.env_configs.policies as policies
from src.environments.env_configs.spaces import (
    ActionSpace,
    ObservationSpace,
    LinearObservation,
    LinearObservationSpaces,
)
from stable_baselines3.common.vec_env import VecEnv
from src.environments.env_configs.policies import ASPolicyVec
from collections.abc import Sequence


class MMVecEnv(gym.Env):
    def __init__(
        self,
        data,
        n_envs=1,  # not used, but needed for compatibility i guess?
        # env params
        capital=1000,
        price_decimals=4,
        tick_size=0.0001,
        inventory_target=0,
        max_order_size=10,
        max_ticks=10,
        reward_class=BaseRewardClass,
        reward_params={},
        record_values=True,
        action_space=ActionSpace.NormalizedAction,
        observation_space=LinearObservation,
        # vectorized env params
        time_envs=1,
        data_portion=0.9,
        inv_envs=1,
        inv_jump=0.25,
        # other params
        column_mapping={},  # which columns are what
        as_expert_params={},
        use_copy_envs=False,
        reset_metrics_on_reset=True,
        **kwargs,
    ):
        self.data = data
        self.capital = capital
        self.price_decimals = price_decimals
        self.tick_size = tick_size
        self.inventory_target = inventory_target
        self.max_order_size = max_order_size
        self.max_diff = tick_size * max_ticks  # max diff between mid and bid/ask
        self.reward_class_type = reward_class
        self.reward_params = reward_params
        self.record_values = record_values
        self.column_mapping = column_mapping

        """
        - time_envs: how many envs with different starting and ending times
            - data_portion: how much of the data is used for each env
        - inv_envs: how many envs with different starting inventory (prefer to be odd number)
            - inv_jump: how much the inventory jumps between envs

        """
        self.use_copy_envs = use_copy_envs
        if not self.use_copy_envs:
            # if inv_envs % 2 == 0:
            #     logging.warning(f"Using {inv_envs} inv_envs, prefer using odd number")
            if (inv_envs - 1) * inv_jump >= 1:
                # TODO: allowed range for norm_inventory is currently [-3,3], this should accomodate
                raise ValueError(
                    f"inv_jump {inv_jump} too large for {inv_envs} inv_envs"
                )
            self.n_envs = time_envs * inv_envs
            data_points = len(data) - 1
            if data_portion * time_envs < 1:
                start = np.array(range(time_envs)) * data_portion
                end = start + data_portion
                start_val = start * data_points
                end_val = end * data_points

            else:
                offset = (1 - data_portion) / (time_envs - 1)
                start = np.array(range(time_envs)) * offset
                end = start + data_portion
                start_val = start * data_points
                end_val = end * data_points
            start_val = start_val.astype(int)
            end_val = end_val.astype(int)

            inv_envs -= 1
            inv_envs_list = [0]
            for i in range(inv_envs):
                mult = int(i / 2 + 1)
                if i % 2 == 0:
                    inv_envs_list.append(inv_jump * mult)
                else:
                    inv_envs_list.append(-inv_jump * mult)

            env_list = list(product(zip(start_val, end_val), inv_envs_list))
            self.start_steps = np.array([i[0][0] for i in env_list])
            self.end_steps = np.array([i[0][1] for i in env_list])
            self.start_inv = np.array([i[1] for i in env_list])
        else:
            self.n_envs = n_envs
            self.start_steps = np.zeros(n_envs).astype(int)
            self.end_steps = np.full(n_envs, len(data) - 1)
            self.start_inv = np.zeros(n_envs)

        # Finished initializing envs

        # to observe the external environment we need best bid and ask and possible trades (price and size), for internal just inventory
        # Everything else depends on the inputs of the policy we are using
        # observations are read from raw data as is but then normalized to be in the given ranges
        self.best_ask = None
        self.best_bid = None
        self.low_price = None
        self.high_price = None
        self.mid_price = None

        # NOTE: OLD
        # self.attribute_list = []
        # for k, v in column_mapping.items():
        #     setattr(self, k, data[:, v])
        #     self.attribute_list.append(k)
        # NOTE: NEW
        # instead of setting attributes individually, have a single external_obs attribute
        self.reset_metrics_on_reset = reset_metrics_on_reset
        self.timestamp = self.data[:, self.column_mapping["timestamp"]]
        self.mid_price = self.data[:, self.column_mapping["mid_price"]]
        self.best_bid = self.data[:, self.column_mapping["best_bid"]]
        self.best_ask = self.data[:, self.column_mapping["best_ask"]]
        self.low_price = self.data[:, self.column_mapping["low_price"]]
        self.high_price = self.data[:, self.column_mapping["high_price"]]
        self.obs_space = observation_space
        if isinstance(self.obs_space, ObservationSpace):
            self.observation_space = self.obs_space.value
        elif isinstance(self.obs_space, LinearObservation):
            if (
                self.obs_space.space_type
                == LinearObservationSpaces.EverythingLinearSpaceAS
            ):
                if as_expert_params == {}:
                    raise ValueError(
                        "Using EverythingLinearSpaceAS but no as_expert_params given"
                    )
            self.observation_space = self.obs_space.obs_space
            self.external_obs = self.data[
                :, [self.column_mapping[i] for i in self.obs_space.external]
            ]

        self.act_space = action_space
        self.action_space = self.act_space.value

        # rewards might use other already initialized parameters from the env, set it here
        self.reward_class = self.reward_class_type(self, **self.reward_params)

        # initialize measured values, can't be done in reset because vecenv are automatically reset
        # this would zero out the values when reaching end of episode
        self.values = []
        self.inventory_values = []
        self.inventory_qty_values = []
        self.limit_buy = []
        self.limit_sell = []
        self.market_buy = []
        self.market_sell = []

        self.trading_bid = []
        self.trading_bid_type = []
        self.trading_bid_hit = []
        self.trading_bid_limit_value = []
        self.trading_bid_market_value = []
        self.trading_bid_limit_volume = []
        self.trading_bid_market_volume = []

        self.trading_ask = []
        self.trading_ask_type = []
        self.trading_ask_hit = []
        self.trading_ask_limit_value = []
        self.trading_ask_market_value = []
        self.trading_ask_limit_volume = []
        self.trading_ask_market_volume = []

        self.trade_market = np.zeros(self.n_envs)
        self.trade_limit = np.zeros(self.n_envs)
        self.spread = np.zeros(self.n_envs)
        self.as_expert_params = as_expert_params
        if self.as_expert_params != {}:
            expert_policy = ASPolicyVec(n_envs=self.n_envs, **as_expert_params)
            self.action_func = expert_policy.get_action_no_mid_no_size

        self.reset()
        return None

    def get_env_params(self):
        return {
            "capital": self.capital,
            "price_decimals": self.price_decimals,
            "tick_size": self.tick_size,
            "inventory_target": self.inventory_target,
            "max_order_size": self.max_order_size,
            "max_diff": self.max_diff,
            "reward_class": self.reward_class_type,
            "reward_params": self.reward_params,
            "record_values": self.record_values,
            "column_mapping": self.column_mapping,
            "action_space": self.act_space,
            "observation_space": self.obs_space,
            "as_expert_params": self.as_expert_params,
            "use_copy_envs": self.use_copy_envs,
        }

    def reset(self):
        self.current_step = self.start_steps.copy()
        # start inventory (normalized)
        i = self.start_inv.copy()
        # cash
        c = np.full(self.n_envs, float(self.capital))
        t = self.inventory_target
        p = self.mid_price[self.current_step]
        self.inventory_qty = -(c * (t + i)) / (p * (t + i - 1))
        # get current price for each env to get all of them to have same total value
        inventory_value = self.inventory_qty * p
        self.quote = self.capital - inventory_value
        # normalized_inventory, this is updated when calling _get_observation, can initialize at 0
        self.norm_inventory = np.zeros(self.n_envs).astype(np.float64)

        self.bids = np.array(0 * self.n_envs).astype(np.float64)
        self.bid_sizes = np.zeros(self.n_envs)
        self.asks = np.zeros(self.n_envs)
        self.ask_sizes = np.zeros(self.n_envs)

        # some environments use metrics to measure performance, other times might want to record values for plotting
        # this flag allows that operation to be turned off
        if self.reset_metrics_on_reset:
            self.reset_metrics()

        return self._get_observation()

    def step(self, action_vec):
        # initialize
        self.spread = np.zeros(self.n_envs)
        self._apply_action(action_vec)
        self.reward_class.start_step()

        # possibly execute the orders
        self._execute_market_orders()
        self._execute_limit_orders()

        # update timeseries variables amd produce observation
        value_end = self._get_value()
        if self.record_values:
            self.values.append(value_end)  # VECTORIZE

        # determine if episode is done
        is_over = self.current_step == self.end_steps

        # increment step unless episode is done
        self.current_step += 1 - is_over
        obs = self._get_observation()

        info = np.repeat({}, self.n_envs)
        if self.record_values:
            self.inventory_qty_values.append(self.inventory_qty.copy())
            self.inventory_values.append(self.norm_inventory)  # VECTORIZE

        # calculate reward (here because lookback values are now saved)
        reward = self.reward_class.end_step()

        # check done
        is_dones = is_over.T
        for i in range(self.n_envs):
            if is_dones[i]:
                info[i] = dict(terminal_observation=obs[i])

        # # logging.debug(f"Reward: {reward}, {reward.shape}")

        return (
            obs,
            reward.reshape(
                self.n_envs,
            ),
            is_dones.reshape(self.n_envs),
            info,
        )

    def _execute_market_orders(self):
        # when doing market orders, we likely lose money from spread
        # buying happens above mid price, selling below mid price

        execute_buy = self.bids >= self.best_ask[self.current_step]
        execute_buy_size = execute_buy * self.bid_sizes
        buy_spread = (self.bids - self.mid_price[self.current_step]) * execute_buy_size
        self.inventory_qty += execute_buy_size
        self.quote -= self.best_ask[self.current_step] * execute_buy_size
        # remove order if executed
        self.bids = self.bids * (1 - execute_buy)
        self.bid_sizes = self.bid_sizes * (1 - execute_buy)
        if self.record_values:
            self.market_buy.append(execute_buy)

        # sell
        execute_sell = self.asks <= self.best_bid[self.current_step]
        execute_sell_size = execute_sell * self.ask_sizes
        sell_spread = (
            self.mid_price[self.current_step] - self.asks
        ) * execute_sell_size
        self.inventory_qty -= execute_sell_size
        self.quote += self.best_bid[self.current_step] * execute_sell_size
        # remove order if executed, means setting ask price to very high number
        self.asks = self.asks * (1 - execute_sell) + execute_sell * 1_000_000
        self.ask_sizes = self.ask_sizes * (1 - execute_sell)
        if self.record_values:
            self.market_sell.append(execute_sell)

        self.trade_market += np.where(execute_buy_size != 0, 1, 0) + np.where(
            execute_sell_size != 0, 1, 0
        )
        self.spread -= sell_spread + buy_spread

    def _execute_limit_orders(self):
        execute_buy = self.bids >= self.low_price[self.current_step]
        execute_buy_size = execute_buy * self.bid_sizes
        buy_spread = (self.mid_price[self.current_step] - self.bids) * execute_buy_size
        self.inventory_qty += execute_buy_size
        self.quote -= self.bids * execute_buy_size
        self.bids = self.bids * (1 - execute_buy)
        self.bid_sizes = self.bid_sizes * (1 - execute_buy)

        execute_sell = self.asks <= self.high_price[self.current_step]
        execute_sell_size = execute_sell * self.ask_sizes
        sell_spread = (
            self.asks - self.mid_price[self.current_step]
        ) * execute_sell_size
        self.inventory_qty -= execute_sell_size
        self.quote += self.asks * execute_sell_size
        self.asks = self.asks * (1 - execute_sell)
        self.ask_sizes = self.ask_sizes * (1 - execute_sell)

        # if self.record_values:
        #     self.limit_buy.append(execute_buy)
        #     self.limit_sell.append(execute_sell)

        # self.trade_limit += np.where(execute_buy_size != 0, 1, 0) + np.where(
        #     execute_sell_size != 0, 1, 0
        # )
        self.spread += sell_spread + buy_spread

    def _get_observation(self):
        # logging.debug(
        #     f"inventory: {self.inventory_qty}, quote: {self.quote}, mid: {self.mid_price[self.current_step]}"
        # )
        self.norm_inventory = np.round(
            self.inventory_qty
            / (self.inventory_qty + self.quote / self.mid_price[self.current_step])
            - self.inventory_target,
            5,
        ).reshape(-1, 1)
        if isinstance(self.obs_space, LinearObservation):
            # val contains the external observations (from file) as determined in the observation space
            val = self.external_obs[self.current_step]

            if (
                self.obs_space.space_type
                == LinearObservationSpaces.EverythingLinearSpace
            ):
                all_vals = np.concatenate((self.norm_inventory, val), axis=1)
                norm = self.obs_space.convert_to_normalized(all_vals)
            elif (
                self.obs_space.space_type
                == LinearObservationSpaces.EverythingLinearSpaceAS
            ):
                # find what AS model would do here, pass inventory, vol and intensity as observations
                vol = self.data[:, self.column_mapping["volatility"]][
                    self.current_step
                ].reshape(-1, 1)
                intensity = self.data[:, self.column_mapping["intensity"]][
                    self.current_step
                ].reshape(-1, 1)

                as_obs = np.concatenate([self.norm_inventory, vol, intensity], axis=1)
                as_action = self.action_func(as_obs)
                as_bid = as_action[:, 0].reshape(-1, 1)
                as_ask = as_action[:, 1].reshape(-1, 1)
                all_vals = np.concatenate(
                    (as_bid, as_ask, self.norm_inventory, val), axis=1
                )
                norm = self.obs_space.convert_to_normalized(all_vals)
            else:
                all_vals = np.concatenate((self.norm_inventory, val), axis=1)
                norm = self.obs_space.convert_to_normalized(all_vals)

            return norm
        else:
            raise NotImplementedError(
                f"Observation space {type(self.obs_space)} not implemented, looking for {LinearObservation}"
            )

    def _apply_action(self, action):
        """
        Depending on the action space, convert the respective action into matching bid/ask sizes and prices
        """
        if action.ndim == 1:
            action = action.reshape(-1, 1).T

        convert_action = {
            ActionSpace.NormalizedAction: policies.convert_continuous_action,
            ActionSpace.NormalizedIntegerAction: policies.convert_integer_action,
            ActionSpace.NoSizeAction: policies.convert_no_mid_no_size_action,
        }

        try:
            formated_action = convert_action[self.act_space](self, action)
        except KeyError:
            raise NotImplementedError(
                f"Conversion not implemented for action_space: {self.act_space}"
            )

        bid_size, ask_size, bid, ask = formated_action
        # NOTE: Used to set this so that the bid/ask size is at least 1
        # make sure both bid and ask size are at least 1
        # bid_size = np.maximum(bid_size, 1)
        # ask_size = np.maximum(ask_size, 1)

        self.bid_sizes, self.ask_sizes, self.bids, self.asks = (
            bid_size,
            ask_size,
            bid,
            ask,
        )
        if self.record_values:
            trading_bid = np.where(bid_size != 0, 1, 0)
            trading_bid_limit = trading_bid * (action[:, 2] < 0)
            trading_bid_market = trading_bid * (action[:, 2] >= 0)
            trading_bid_limit_hit = trading_bid_limit * (
                self.bids >= self.low_price[self.current_step]
            )
            trading_bid_limit_volume = trading_bid_limit_hit * self.bid_sizes
            trading_bid_market_volume = trading_bid_market * self.bid_sizes
            trading_bid_limit_value = trading_bid_limit_volume * (
                self.mid_price[self.current_step] - self.bids
            )
            trading_bid_market_value = -(
                trading_bid_market_volume
                * (self.best_ask[self.current_step] - self.mid_price[self.current_step])
            )

            self.trading_bid.append(trading_bid)
            self.trading_bid_type.append(trading_bid_limit)
            self.trading_bid_hit.append(trading_bid_limit_hit)
            self.trading_bid_limit_volume.append(trading_bid_limit_volume)
            self.trading_bid_market_volume.append(trading_bid_market_volume)
            self.trading_bid_limit_value.append(trading_bid_limit_value)
            self.trading_bid_market_value.append(trading_bid_market_value)

            # print(f"trading bid: {trading_bid}")
            # print(f"trading_bid_limit: {trading_bid_limit}")
            # print(f"trading_bid_limit_hit: {trading_bid_limit_hit}")
            # print(f"trading_bid_limit_value: {trading_bid_limit_value}")
            # print(f"trading_bid_market_value: {trading_bid_market_value}")

            trading_ask = ask_size.astype(bool)
            trading_ask_limit = trading_ask * (action[:, 3] > 0)
            trading_ask_market = trading_ask * (action[:, 3] <= 0)
            trading_ask_limit_hit = trading_ask_limit * (
                self.asks <= self.high_price[self.current_step]
            )
            trading_ask_limit_volume = trading_ask_limit_hit * self.ask_sizes
            trading_ask_market_volume = trading_ask_market * self.ask_sizes
            trading_ask_limit_value = trading_ask_limit_volume * (
                self.asks - self.mid_price[self.current_step]
            )
            trading_ask_market_value = -(
                trading_ask_market_volume
                * (self.mid_price[self.current_step] - self.best_bid[self.current_step])
            )

            self.trading_ask.append(trading_ask)
            self.trading_ask_type.append(trading_ask_limit)
            self.trading_ask_hit.append(trading_ask_limit_hit)
            self.trading_ask_limit_value.append(trading_ask_limit_value)
            self.trading_ask_market_value.append(trading_ask_market_value)
            self.trading_ask_limit_volume.append(trading_ask_limit_volume)
            self.trading_ask_market_volume.append(trading_ask_market_volume)
            # print(f"trading ask: {trading_ask}")
            # print(f"trading_ask_limit: {trading_ask_limit}")
            # print(f"trading_ask_limit_hit: {trading_ask_limit_hit}")
            # print(f"trading_ask_limit_value: {trading_ask_limit_value}")
            # print(f"trading_ask_market_value: {trading_ask_market_value}")

    def _get_value(self):
        """
        How much quote and inventory is worth at the current best bid
        """
        return (
            self.quote + self.inventory_qty * self.best_bid[self.current_step]
        ).reshape(-1, 1)

    def get_raw_recorded_values(self):
        """
        Returns the raw arrays of recorded values
        """
        return {
            "values": self.values,
            "inventory_values": self.inventory_values,
            "inventory_qty": self.inventory_qty_values,
            # bid related
            "trading_bid": self.trading_bid,
            "trading_bid_type": self.trading_bid_type,
            "trading_bid_hit": self.trading_bid_hit,
            "trading_bid_limit_value": self.trading_bid_limit_value,
            "trading_bid_market_value": self.trading_bid_market_value,
            "trading_bid_limit_volume": self.trading_bid_limit_volume,
            "trading_bid_market_volume": self.trading_bid_market_volume,
            # ask related
            "trading_ask": self.trading_ask,
            "trading_ask_type": self.trading_ask_type,
            "trading_ask_hit": self.trading_ask_hit,
            "trading_ask_limit_value": self.trading_ask_limit_value,
            "trading_ask_market_value": self.trading_ask_market_value,
            "trading_ask_limit_volume": self.trading_ask_limit_volume,
            "trading_ask_market_volume": self.trading_ask_market_volume,
        }

    def get_recorded_values_to_df(self):
        raw_values = self.get_raw_recorded_values()
        df_dict = {
            "values": np.array(raw_values["values"]).reshape(1, -1)[0],
            "inventory_values": np.array(raw_values["inventory_values"]).reshape(1, -1)[
                0
            ],
            "inventory_qty": np.array(raw_values["inventory_qty"]).reshape(1, -1)[0],
        }
        return pd.DataFrame(df_dict)

    def get_metrics(self):
        """
        Returns a dictionary of precalculated metrics
        timesteps: integer
        other metrics: arrays of lenght n_envs
        inventory represents normalized inventory, roughly (inventory / (inventory + cash))
        """
        assert self.record_values, "Must set record_values to True to get metrics"
        values = np.array(self.values).reshape(-1, self.n_envs)
        total_return = values[-1, :] / values[0, :] - 1  # VECTORIZE
        drawdown = (values / np.maximum.accumulate(values, axis=0) - 1).min(axis=0)
        returns = np.diff(values, axis=0) / values[:-1, :]
        volatility = np.std(returns, axis=0)
        # volatility = np.std(np.diff(np.array(values)[:, 0]))  # VECTORIZE

        sharpe = total_return / (volatility + 1e-8)
        trades = self.trade_market + self.trade_limit
        max_inventory = np.max(np.abs(self.inventory_values), axis=0)
        values = {
            "timesteps": len(values),
            "episode_return": total_return,
            "sharpe": sharpe,
            "drawdown": drawdown,
            "trades": trades,
            # "limit_trades": self.trade_limit,
            # "market_trades": self.trade_market,
            "max_inventory": max_inventory,
            "mean_abs_inv": np.mean(np.abs(self.inventory_values), axis=0),
            # bid
            "bid_trade_count": np.sum(self.trading_bid, axis=0),
            "bid_limit_count": np.sum(self.trading_bid_type, axis=0),
            "bid_limit_hit_count": np.sum(self.trading_bid_hit, axis=0),
            "bid_limit_value": np.sum(self.trading_bid_limit_value, axis=0),
            "bid_market_value": np.sum(self.trading_bid_market_value, axis=0),
            "bid_limit_volume": np.sum(self.trading_bid_limit_volume, axis=0),
            "bid_market_volume": np.sum(self.trading_bid_market_volume, axis=0),
            # ask
            "ask_trade_count": np.sum(self.trading_ask, axis=0),
            "ask_limit_count": np.sum(self.trading_ask_type, axis=0),
            "ask_limit_hit_count": np.sum(self.trading_ask_hit, axis=0),
            "ask_limit_value": np.sum(self.trading_ask_limit_value, axis=0),
            "ask_market_value": np.sum(self.trading_ask_market_value, axis=0),
            "ask_limit_volume": np.sum(self.trading_ask_limit_volume, axis=0),
            "ask_market_volume": np.sum(self.trading_ask_market_volume, axis=0),
        }
        return values

    def get_metrics_single(self):
        assert self.n_envs == 1, "Only implemented for n_envs = 1"
        metrics = self.get_metrics()
        for k, v in metrics.items():
            new_val = _check_nested_array(v)
            metrics[k] = new_val  # or whatever you want to do with iterable items
        return metrics

    def get_metrics_val(self):
        if self.n_envs != 1:
            raise NotImplementedError(
                " 'get_metrics_val' Only implemented for n_envs = 1"
            )
        metrics = self.get_metrics()
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                metrics[k] = v[0]
        return metrics

    def get_metrics_str(self):
        values = self.get_metrics()
        str_rep = ""
        for k, v in values.items():
            if isinstance(v, np.ndarray):
                v = v.tolist()
            str_rep += f"{k}: {v}\n"
        return str_rep

    def save_metrics(self, path):
        data_dict = {
            "value": np.array(self.values)[:, 0],
            "inventory": np.array(self.inventory)[:, 0],
        }
        pd.DataFrame(data_dict).to_csv(
            os.path.join(os.getenv("RESULT_PATH"), f"{path}.csv"), index=False
        )

    def reset_metrics(self):
        # if reset
        self.values = []
        self.inventory_values = []
        self.inventory_qty_values = []
        self.trade_market = np.zeros(self.n_envs)
        self.trade_limit = np.zeros(self.n_envs)

        self.limit_buy = []
        self.limit_sell = []
        self.market_buy = []
        self.market_sell = []

        self.trading_bid = []
        self.trading_bid_type = []
        self.trading_bid_hit = []
        self.trading_bid_limit_value = []
        self.trading_bid_market_value = []
        self.trading_bid_limit_volume = []
        self.trading_bid_market_volume = []

        self.trading_ask = []
        self.trading_ask_type = []
        self.trading_ask_hit = []
        self.trading_ask_limit_value = []
        self.trading_ask_market_value = []
        self.trading_ask_limit_volume = []
        self.trading_ask_market_volume = []


def _check_nested_array(maybe_arr):
    if isinstance(maybe_arr, np.ndarray):
        return _check_nested_array(maybe_arr[0])
    else:
        return maybe_arr


class SBMMVecEnv(VecEnv):
    def __init__(self, env, **kwargs):
        self.env = env
        self.reset_envs = True
        self.actions: np.ndarray = self.env.action_space.sample()
        super().__init__(
            self.env.n_envs, self.env.observation_space, self.env.action_space
        )

    def reset(self):
        return self.env.reset()

    def action_masks(self):
        return self.env.action_masks()

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self):
        state, rewards, dones, infos = self.env.step(self.actions)
        if dones.min() and self.reset_envs:
            state = (
                self.env.reset()
            )  # StableBaselines VecEnvs need to automatically reset themselves.
        return state, rewards, dones, infos

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices=None):
        pass

    def set_attr(self, attr_name: str, value, indices=None):
        pass

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices=None,
        **method_kwargs,
    ):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in range(self.env.n_envs)]

    def seed(self, seed=None):
        self.env.seed(seed)

    def clone_venv(
        self,
        data=None,
        time_envs=1,
        inv_envs=1,
        data_portion=0.9,
        inv_jump=0.25,
    ):
        """
        Returns the same environment that is completely detached from the current one.
        """
        if data is None:
            data = self.env.data.copy()
        env_clone = MMVecEnv(
            data=data,
            time_envs=time_envs,
            inv_envs=inv_envs,
            data_portion=data_portion,
            inv_jump=inv_jump,
            **self.env.get_env_params(),
        )
        return SBMMVecEnv(env_clone)
