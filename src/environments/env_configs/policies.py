from abc import ABC
import numpy as np

from environments.env_configs.spaces import (
    ActionSpace,
    ObservationSpace,
    LinearObservation,
)


class PolicyBase(ABC):
    def __init__(
        self,
        max_order_size,
        tick_size,
        max_ticks,
        price_decimals,
        observation_space,
        action_space,
    ):
        self.max_order_size = max_order_size
        self.tick_size = tick_size
        self.max_ticks = max_ticks
        self.price_decimals = price_decimals
        self.observation_space = observation_space
        self.action_space = action_space

    def get_action(self, state):
        raise NotImplementedError

    def get_action_space(self):
        return self.action_space.value

    def get_observation_space(self):
        return self.observation_space.value


class ASPolicyVec:
    def __init__(
        self,
        env,
        max_order_size,
        tick_size,
        max_ticks,
        price_decimals,
        inventory_target,
        risk_aversion,
        order_size,
        max_diff=0.001,
        obs_type=ObservationSpace.OSIObservation,
        act_type=ActionSpace.NormalizedAction,
    ):
        self.env = env
        self.max_order_size = max_order_size
        self.tick_size = tick_size
        self.max_diff = max_diff
        self.max_ticks = max_ticks
        self.price_decimals = price_decimals
        self.inventory_target = inventory_target
        self.risk_aversion = risk_aversion
        self.order_size = np.full(self.env.n_envs, order_size)
        self.obs_type = obs_type
        self.act_type = act_type

    def get_continuous_action(self, observation):
        """
        Continuous action returning:
        - bid size as a percentage of max order size
        - ask size as a percentage of max order size
        - bid price as HALF TICKS from mid price / max half ticks
        - ask price as HALF TICKS from mid price / max half ticks
        """

        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        mid_price = observation[:, 0]
        inventory = observation[:, 1]
        intensity = observation[:, 2]
        vol = observation[:, 3]

        reservation_price = mid_price - inventory * vol * self.risk_aversion
        spread = vol * self.risk_aversion + 2 / self.risk_aversion * np.log(
            1 + self.risk_aversion / intensity
        )

        bid_size = self.order_size
        ask_size = self.order_size
        bid_size_normalize = bid_size / self.max_order_size
        ask_size_normalize = ask_size / self.max_order_size

        # bid and ask prices (actual)
        bid = np.round(reservation_price - spread / 2, self.price_decimals)
        bid_half_ticks = ((bid - mid_price) / self.tick_size) / 2
        bid_tick_ratio = bid_half_ticks / self.max_ticks

        ask = np.round(reservation_price + spread / 2, self.price_decimals)
        ask_half_ticks = ((ask - mid_price) / self.tick_size) / 2
        ask_tick_ratio = ask_half_ticks / self.max_ticks

        return np.array(
            [bid_size_normalize, ask_size_normalize, bid_tick_ratio, ask_tick_ratio]
        ).T

    def get_action_no_mid(self, observation):
        """
        Alternative action space that does not use mid price. Otherwise same as continuous action

        """
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        bid_size = self.order_size
        ask_size = self.order_size

        bid_size_normalize = np.full(self.env.n_envs, bid_size / self.max_order_size)
        ask_size_normalize = np.full(self.env.n_envs, ask_size / self.max_order_size)

        inventory = observation[:, 0]
        vol = observation[:, 1]
        intensity = observation[:, 2]

        spread = vol * self.risk_aversion + 2 / self.risk_aversion * np.log(
            1 + self.risk_aversion / intensity
        )
        inventory_adjustment = -inventory * vol * self.risk_aversion

        # relative USD amount from mid price
        bid = inventory_adjustment - spread / 2
        # half ticks from mid price
        # bid_ticks = np.round((bid / self.tick_size) * 2, 0)
        # normalized w.r.t. max ticks
        bid_normalize = bid / self.max_diff

        ask = inventory_adjustment + spread / 2
        # ask_ticks = np.round((ask / self.tick_size) * 2, 0)
        ask_normalize = ask / self.max_diff

        return np.array(
            [bid_size_normalize, ask_size_normalize, bid_normalize, ask_normalize]
        ).T

    def get_action_no_mid_no_size(self, observation):
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        inventory = observation[:, 0]
        vol = observation[:, 1]
        intensity = observation[:, 2]
        spread = vol * self.risk_aversion + 2 / self.risk_aversion * np.log(
            1 + self.risk_aversion / intensity
        )
        inventory_adjustment = -inventory * vol * self.risk_aversion

        bid = inventory_adjustment - spread / 2
        bid_normalize = bid / self.max_diff

        ask = inventory_adjustment + spread / 2
        ask_normalize = ask / self.max_diff

        return np.array([bid_normalize, ask_normalize]).T

    def get_no_size_action_linear(self, observation):
        # the input is normalized
        obs_dict = {
            "inventory": observation[:, 0],
            "volatility": observation[:, 1],
            "intensity": observation[:, 2],
        }
        obs_dict_converted = self.obs_type.convert_to_readable(obs_dict)
        obs = []
        for i in ["inventory", "volatility", "intensity"]:
            obs.append(obs_dict_converted[i])
        return self.get_action_no_mid_no_size(np.array(obs).T)

    def get_action_func(self):
        if type(self.obs_type) == ObservationSpace:
            action_func_dict = {
                (
                    ObservationSpace.OSIObservation,
                    ActionSpace.NormalizedAction,
                ): self.get_continuous_action,
                (
                    ObservationSpace.SimpleObservation,
                    ActionSpace.NormalizedAction,
                ): self.get_action_no_mid,
                (
                    ObservationSpace.SimpleObservation,
                    ActionSpace.NoSizeAction,
                ): self.get_action_no_mid_no_size,
                (
                    LinearObservation,
                    ActionSpace.NoSizeAction,
                ): self.get_no_size_action_linear,
            }
            key = (self.obs_type, self.act_type)
        elif type(self.obs_type) == LinearObservation:
            action_func_dict = {
                ActionSpace.NoSizeAction: self.get_no_size_action_linear,
            }
            key = self.act_type

        try:
            func = action_func_dict[key]
            return func
        except KeyError:
            raise NotImplementedError(
                f"action func not implemented for this obs/act combination: {input}"
            )


def convert_continuous_action(env, action):
    # this is not fully continuous but uses float and looks like one
    bid_sizes = np.full(env.n_envs, 1)
    ask_sizes = np.full(env.n_envs, 1)

    bid_diff = action[:, 2] * env.max_diff
    bid_price = env.mid_price[env.current_step] + bid_diff
    bid_round = np.round(bid_price, env.price_decimals)

    ask_diff = action[:, 3] * env.max_diff
    ask_price = env.mid_price[env.current_step] + ask_diff
    ask_round = np.round(ask_price, env.price_decimals)

    # asks = action[:, 3] * env.max_ticks * env.tick_size
    # bids_round = np.round(env.mid_price[env.current_step] + bids, env.price_decimals)
    # asks_round = np.round(env.mid_price[env.current_step] + asks, env.price_decimals)
    return bid_sizes, ask_sizes, bid_round, ask_round


def convert_integer_action(env, action):
    # here we convert the integer action produced by agent to similar numbers as continuous action
    # to allow for env to execute the action
    action = np.round(action)
    bid_sizes = np.full(env.n_envs, 1)
    ask_sizes = np.full(env.n_envs, 1)

    bids = np.round(
        action[:, 2] * (env.tick_size / 2) + env.mid_price[env.current_step],
        env.price_decimals,
    )
    asks = np.round(
        action[:, 3] * (env.tick_size / 2) + env.mid_price[env.current_step],
        env.price_decimals,
    )
    return bid_sizes, ask_sizes, bids, asks


def convert_no_mid_no_size_action(env, action):
    bids = np.round(
        (action[:, 0] * env.max_diff) + env.mid_price[env.current_step],
        env.price_decimals,
    )
    asks = np.round(
        (action[:, 1] * env.max_diff) + env.mid_price[env.current_step],
        env.price_decimals,
    )
    return np.full(env.n_envs, 1), np.full(env.n_envs, 1), bids, asks
