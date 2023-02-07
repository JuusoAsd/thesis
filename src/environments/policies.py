import logging
from abc import ABC


import numpy as np
from gym import spaces


class PolicyBase(ABC):
    def __init__(self, env, params):
        self.env = env
        for k, v in params.items():
            setattr(self, k, v)

    def get_action_space(self):
        raise NotImplementedError

    def get_action(self, state):
        # Not required for ML agents
        raise NotImplementedError

    def apply_action(self, action):
        raise NotImplementedError

    def get_observation():
        raise NotImplementedError

    def get_observation_space(self, required_observations):
        raise NotImplementedError


class AvellanedaStoikovPolicy(PolicyBase):
    """
    Policy that determines bid and ask based on the Avellaneda-Stoikov model
    params:
        risk_aversion: risk aversion parameter
        inventory_target: inventory target parameter
        order_size: size of orders to place
    returns:
        array of bid size, ask size, bid price, ask price
    """

    def __init__(self, env, risk_aversion, order_size, **params):
        super().__init__(env, params)
        self.risk_aversion = risk_aversion
        self.order_size = order_size

    def get_action_space(self):
        action_dict = {
            "bid_size": [0, 1000],
            "ask_size": [0, 1000],
            "bid": [0, 1000],
            "ask": [0, 1000],
        }
        return spaces.Box(
            low=np.array([action_dict[key][0] for key in action_dict.keys()]),
            high=np.array([action_dict[key][1] for key in action_dict.keys()]),
            shape=(len(action_dict.keys()),),
            dtype=np.float32,
        )

    def get_action(self):
        mid_price = round(self.env.current_state.mid_price, self.env.price_decimals)
        vol = self.env.current_state.vol
        intensity = self.env.current_state.intensity
        inventory = self.env.get_inventory(mid_price)

        reservation_price = mid_price - inventory * vol * self.risk_aversion
        spread = vol * self.risk_aversion + 2 / self.risk_aversion * np.log(
            1 + self.risk_aversion / intensity
        )

        bid_size = self.order_size
        ask_size = self.order_size
        bid = round(reservation_price - spread / 2, self.env.price_decimals)
        ask = round(reservation_price + spread / 2, self.env.price_decimals)

        if self.env.logging:
            self.env.logger.info(
                f"update,{self.env.current_state.timestamp},{self.env.bid},{self.env.ask},{mid_price},{vol},{intensity},{inventory},{reservation_price},{spread}"
            )
        logging.debug(
            f"mid price: {mid_price}, bid: {self.env.bid}, ask: {self.env.ask}, inventory: {inventory}, vol: {vol}, intensity: {intensity}"
        )
        return np.array([bid_size, ask_size, bid, ask])

    def apply_action(self, action):
        self.env.bid_size = action[0]
        self.env.ask_size = action[1]
        self.env.bid = action[2]
        self.env.ask = action[3]

    def get_observation_space(self, required_observations):
        required_observations["volatility"] = [0, 100_000]
        required_observations["intensity"] = [0, 100_000]
        return spaces.Box(
            low=np.array(
                [required_observations[key][0] for key in required_observations.keys()]
            ),
            high=np.array(
                [required_observations[key][1] for key in required_observations.keys()]
            ),
            shape=(len(required_observations.keys()),),
            dtype=np.float64,
        )

    def get_observation(self):
        best_bid = self.env.current_state.best_bid
        best_ask = self.env.current_state.best_ask
        mid_price = (best_bid + best_ask) / 2
        return np.array(
            [
                best_bid,
                best_ask,
                self.env.get_inventory(mid_price),
                self.env.current_state.vol,
                self.env.current_state.intensity,
            ],
            dtype=np.float64,
        )


class MLPolicy(PolicyBase):
    def __init__(self, env, max_size, **params):
        super().__init__(env, params)
        self.max_size = max_size

    def get_action_space(self):
        # a normalized action space
        action_dict = {
            "bid_size": [0, 1],  # bid size from 0 to max bid
            "ask_size": [0, 1],
            "bid": [-1, 1],  # bid price from min to max in ticks from mid price
            "ask": [-1, 1],
        }
        return spaces.Box(
            low=np.array([action_dict[key][0] for key in action_dict.keys()]),
            high=np.array([action_dict[key][1] for key in action_dict.keys()]),
            shape=(len(action_dict.keys()),),
            dtype=np.float32,
        )

    def denormalize_action(self, action):
        def _denormalize_size(size):
            return round(size * self.max_size, self.env.price_decimals)

        def _denormalize_price(ticks):
            mid_price = self.env.current_state.mid_price
            return round(
                mid_price + ticks * self.env.tick_size, self.env.price_decimals
            )

        bid_size = _denormalize_size(action[0])
        ask_size = _denormalize_size(action[1])
        bid = _denormalize_price(action[2])
        ask = _denormalize_price(action[3])
        return [bid_size, ask_size, bid, ask]

    def apply_action(self, action):
        denorm_action = self.denormalize_action(action)
        self.env.bid_size = denorm_action[0]
        self.env.ask_size = denorm_action[1]
        self.env.bid = denorm_action[2]
        self.env.ask = denorm_action[3]

    def get_observation_space(self, required_observations):
        required_observations["volatility"] = [0, 100_000]
        required_observations["intensity"] = [0, 100_000]
        return spaces.Box(
            low=np.array(
                [required_observations[key][0] for key in required_observations.keys()]
            ),
            high=np.array(
                [required_observations[key][1] for key in required_observations.keys()]
            ),
            shape=(len(required_observations.keys()),),
            dtype=np.float64,
        )

    def get_observation(self):
        best_bid = self.env.current_state.best_bid
        best_ask = self.env.current_state.best_ask
        mid_price = (best_bid + best_ask) / 2
        return np.array(
            [
                best_bid,
                best_ask,
                self.env.get_inventory(mid_price),
                self.env.current_state.vol,
                self.env.current_state.intensity,
            ],
            dtype=np.float64,
        )
