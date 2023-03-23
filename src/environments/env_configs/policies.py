from abc import ABC
import numpy as np

from environments.env_configs.spaces import ActionSpace, ObservationSpace


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

    def normalize(self, action, mid_price):
        bid_size = action[0]
        ask_size = action[1]
        bid = action[2]
        ask = action[3]

        bid_size_normalize = bid_size / self.max_order_size
        ask_size_normalize = ask_size / self.max_order_size
        bid_normalize = ((bid - mid_price) / self.tick_size) / self.max_ticks
        ask_normalize = ((ask - mid_price) / self.tick_size) / self.max_ticks

        return np.array(
            [[bid_size_normalize, ask_size_normalize, bid_normalize, ask_normalize]]
        )

    def get_action(self, state):
        raise NotImplementedError

    def get_action_space(self):
        return self.action_space.value

    def get_observation_space(self):
        return self.observation_space.value


class AvellanedaStoikovPolicy(PolicyBase):
    """
    Policy that determines bid and ask based on the Avellaneda-Stoikov model
    params:
        order_size: size of orders to place
        inventory_target: inventory target parameter
        risk_aversion: risk aversion parameter
    returns:
        array of bid size, ask size, bid price, ask price
    """

    def __init__(
        self,
        max_order_size,
        tick_size,
        max_ticks,
        price_decimals,
        observation_space,
        action_space,
        order_size,
        # inventory_target,
        risk_aversion,
    ):
        super().__init__(
            max_order_size,
            tick_size,
            max_ticks,
            price_decimals,
            observation_space,
            action_space,
        )
        self.order_size = order_size
        # self.inventory_target = inventory_target
        self.risk_aversion = risk_aversion
        self.observation_space = ObservationSpace.ASObservation.value
        self.action_space = ActionSpace.NormalizedAction.value

    def get_action(self, obs):
        obs = obs[0]
        best_bid = obs[0]
        best_ask = obs[1]
        inventory = obs[2]
        vol = obs[3]
        intensity = obs[4]
        mid_price = (best_bid + best_ask) / 2
        reservation_price = mid_price - inventory * vol * self.risk_aversion
        spread = vol * self.risk_aversion + 2 / self.risk_aversion * np.log(
            1 + self.risk_aversion / intensity
        )

        bid_size = self.order_size
        ask_size = self.order_size
        bid = round(reservation_price - spread / 2, self.price_decimals)
        ask = round(reservation_price + spread / 2, self.price_decimals)

        return self.normalize(np.array([bid_size, ask_size, bid, ask]), mid_price)


class ASPolicyVec:
    def __init__(
        self,
        max_order_size,
        tick_size,
        max_ticks,
        price_decimals,
        inventory_target,
        risk_aversion,
        order_size,
        n_env=1,
        obs_type=ObservationSpace.OSIObservation,
        act_type=ActionSpace.NormalizedAction,
    ):
        self.max_order_size = max_order_size
        self.tick_size = tick_size
        self.max_ticks = max_ticks
        self.price_decimals = price_decimals
        self.inventory_target = inventory_target
        self.risk_aversion = risk_aversion
        self.order_size = np.full(n_env, order_size)
        self.n_env = n_env
        self.obs_type = obs_type
        self.act_type = act_type

    def get_continuous_action(self, observation):
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
        bid = np.round(reservation_price - spread / 2, self.price_decimals)
        ask = np.round(reservation_price + spread / 2, self.price_decimals)
        bid_size_normalize = bid_size / self.max_order_size
        ask_size_normalize = ask_size / self.max_order_size
        bid_normalize = ((bid - mid_price) / self.tick_size) / self.max_ticks
        ask_normalize = ((ask - mid_price) / self.tick_size) / self.max_ticks

        return np.array(
            [bid_size_normalize, ask_size_normalize, bid_normalize, ask_normalize]
        ).T

    def get_integer_action(self, observation):
        # returns same as action float but using integers
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

        bid = np.round(reservation_price - spread / 2, self.price_decimals)
        ask = np.round(reservation_price + spread / 2, self.price_decimals)

        # use half ticks to normalize the size, mid price usually in middle of tick
        bid_normalize = np.round(((bid - mid_price) / (self.tick_size / 2))).astype(int)
        ask_normalize = np.round(((ask - mid_price) / (self.tick_size / 2))).astype(int)
        return np.array([bid_size, ask_size, bid_normalize, ask_normalize]).T

    def get_action_no_mid(self, observation):
        """
        Alternative action space that does not use mid price
        mid price fluctuates and might be difficult to detect that it is actually sort of irrelevant
        """
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        inventory = observation[:, 0]
        vol = observation[:, 1]
        intensity = observation[:, 2]

        spread = vol * self.risk_aversion + 2 / self.risk_aversion * np.log(
            1 + self.risk_aversion / intensity
        )
        inventory_adjustment = -inventory * vol * self.risk_aversion
        bid_size = self.order_size
        ask_size = self.order_size

        bid_size_normalize = bid_size / self.max_order_size
        ask_size_normalize = ask_size / self.max_order_size

        bid = np.round(inventory_adjustment - spread / 2, self.price_decimals)
        ask = np.round(inventory_adjustment + spread / 2, self.price_decimals)

        bid_normalize = (bid / self.tick_size) / self.max_ticks
        ask_normalize = (ask / self.tick_size) / self.max_ticks

        return np.array(
            [bid_size_normalize, ask_size_normalize, bid_normalize, ask_normalize]
        ).T

    def get_action_func(self):
        if self.obs_type == ObservationSpace.OSIObservation:
            if self.act_type == ActionSpace.NormalizedAction:
                return self.get_continuous_action
            elif self.act_type == ActionSpace.IntegerAction:
                return self.get_integer_action
        elif self.obs_type == ObservationSpace.SimpleObservation:
            if self.act_type == ActionSpace.NormalizedAction:
                return self.get_action_no_mid
        raise NotImplementedError(
            "action func not implemented for this obs/act combination"
        )


class AlwaysSamePolicyVec:
    def __init__(self, env) -> None:
        self.env = env

    def get_action(self, observation):
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        # returns same shape action as before to make things easier
        constant_value = 0.5
        a = np.full(self.env.n_envs, constant_value)
        b = np.full(self.env.n_envs, constant_value)
        c = np.full(self.env.n_envs, constant_value)
        d = np.full(self.env.n_envs, constant_value)

        return np.array([a, b, c, d]).T

    def get_action_func(self):
        return self.get_action


class SimplePolicyVec:
    """
    returns:
        - absolute value of inventory
        - inventory / 2
        - volatility / 2
        - volatility ** 2
    """

    def __init__(self, env, **kwargs) -> None:
        self.env = env

    def get_action(self, observation):
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        # numbers can be very large, need to normalize to match action space
        # Can't be a constant function tho
        # inventory: [-1, 1]
        # volatility: [0, 1]
        # intensity: [0, 100_000]
        # want to fit actions on [0, 1]

        # do not use inventory, might be that BC won't see different inventory states?
        # inventory = observation[:, 0]
        vol = observation[:, 1]

        # normalize < 1
        intensity = observation[:, 2] / 100_000

        a = np.abs(intensity)
        b = np.abs(intensity) / 2
        c = vol / 2
        d = vol**2

        return np.array([a, b, c, d]).T

    def get_action_func(self):
        return self.get_action


def convert_continuous_action(env, action):
    # this is not fully continuous but uses float and looks like one

    bid_sizes = np.full(env.n_envs, 1)
    ask_sizes = np.full(env.n_envs, 1)

    bids = action[:, 2] * env.max_ticks * env.tick_size
    asks = action[:, 3] * env.max_ticks * env.tick_size
    bids_round = np.round(env.mid_price[env.current_step] + bids, env.price_decimals)
    asks_round = np.round(env.mid_price[env.current_step] + asks, env.price_decimals)

    return bid_sizes, ask_sizes, bids_round, asks_round


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
