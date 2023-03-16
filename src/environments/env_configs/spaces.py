from enum import Enum

import numpy as np
from gym import spaces


def get_space(space_dict):
    return spaces.Box(
        low=np.array([space_dict[key][0] for key in space_dict.keys()]),
        high=np.array([space_dict[key][1] for key in space_dict.keys()]),
        shape=(len(space_dict.keys()),),
        dtype=np.float32,
    )


class ActionSpace(Enum):
    # Enum for different possible action spaces
    NormalizedAction = get_space(
        {
            "bid_size": [0, 1],  # bid size from 0 to max bid
            "ask_size": [0, 1],
            "bid": [-1, 1],  # bid price from min to max in ticks from mid price
            "ask": [-1, 1],
        }
    )


class ObservationSpace(Enum):
    # Enum for different possible observation spaces
    ASObservation = (
        get_space(
            {
                "best_bid": [0, 1000],
                "best_ask": [0, 1000],
                "inventory": [-1, 1],
                "volatility": [0, 100_000],
                "intensity": [0, 100_000],
            }
        ),
    )
    OSIObservation = get_space(
        {
            "mid_price": [0, 1000],
            "inventory": [-1, 1],
            "volatility": [0, 100_000],
            "intensity": [0, 100_000],
            "osi": [-100, 100],
        }
    )


def get_observation(env):
    if env.params["observation_space"] == ObservationSpace.ASObservation:
        return np.array(
            [
                env.current_state.best_bid,
                env.current_state.best_ask,
                env.get_inventory(env.current_state.mid_price),
                env.current_state.volatility,
                env.current_state.intensity,
            ]
        )
    else:
        raise ValueError("Invalid observation space")


def apply_action(env, action):
    if env.params["action_space"] == ActionSpace.NormalizedAction:
        bid_size = action[0] * env.policy_params["max_order_size"]
        ask_size = action[1] * env.policy_params["max_order_size"]
        bid = (
            action[2] * env.policy_params["max_ticks"] * env.policy_params["tick_size"]
        )
        ask = (
            action[3] * env.policy_params["max_ticks"] * env.policy_params["tick_size"]
        )

        bid_price = round(env.current_state.mid_price + bid, env.price_decimals)
        ask_price = round(env.current_state.mid_price + ask, env.price_decimals)
        env.bid_size = bid_size
        env.ask_size = ask_size
        env.bid_price = bid_price
        env.ask_price = ask_price
