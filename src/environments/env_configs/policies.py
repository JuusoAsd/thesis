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
    ):
        self.max_order_size = max_order_size
        self.tick_size = tick_size
        self.max_ticks = max_ticks
        self.price_decimals = price_decimals
        self.inventory_target = inventory_target
        self.risk_aversion = risk_aversion
        self.order_size = np.full(n_env, order_size)
        self.n_env = n_env

    def get_action(self, observation):
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

    def get_action_func(self):
        return self.get_action
