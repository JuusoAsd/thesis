from enum import Enum

import numpy as np
from gym import spaces


def get_space(space_dict):
    return spaces.Box(
        low=np.float32(np.array([space_dict[key][0] for key in space_dict.keys()])),
        high=np.float32(np.array([space_dict[key][1] for key in space_dict.keys()])),
        shape=(len(space_dict.keys()),),
        dtype=np.float64,
    )


def get_integer_space(space_dict):
    return spaces.Box(
        low=np.float32(np.array([space_dict[key][0] for key in space_dict.keys()])),
        high=np.float32(np.array([space_dict[key][1] for key in space_dict.keys()])),
        shape=(len(space_dict.keys()),),
        dtype=int,
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
    # Same as above but instead of using floats uses integers,
    # size is order size as number of units
    # price is number of HALF-ticks from mid price (1 = 0.5 ticks)
    NormalizedIntegerAction = get_integer_space(
        {
            "bid_size": [0, 10],
            "ask_size": [0, 10],
            "bid": [-20, 20],
            "ask": [-20, 20],
        }
    )
    NoSizeAction = get_space(
        {
            "bid": [-1, 1],
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
    SimpleObservation = get_space(
        {
            "inventory": [-1, 1],
            "volatility": [0, 1],
            "intensity": [0, 100_000],
        }
    )
    DummyObservation = get_space(
        {
            "inventory": [-1, 1],
            "volatility": [-1, 1],
            "intensity": [0, 1],
        }
    )


class LinearObservationSpaces(Enum):
    # enum for different possible observation spaces
    # keys must be found in the environment
    OnlyInventorySpace = {
        "inventory": {"min": -1, "max": 1, "min_actual": -2, "max_actual": 2},
    }
    SimpleLinearSpace = {
        "inventory": {"min": -1, "max": 1, "min_actual": -2, "max_actual": 2},
        "volatility": {"min": -1, "max": 1, "min_actual": 0, "max_actual": 0.01},
        "intensity": {"min": -1, "max": 1, "min_actual": 0, "max_actual": 100_000},
    }

    OSILinearSpace = {
        "inventory": {"min": -1, "max": 1, "min_actual": -2, "max_actual": 2},
        "volatility": {"min": -1, "max": 1, "min_actual": 0, "max_actual": 0.01},
        "intensity": {"min": -1, "max": 1, "min_actual": 0, "max_actual": 100_000},
        "osi": {"min": -1, "max": 1, "min_actual": -100, "max_actual": 100},
    }

    EverythingLinearSpace = {
        "inventory": {"min": -1, "max": 1, "min_actual": -2, "max_actual": 2},
        "volatility": {"min": -1, "max": 1, "min_actual": 0, "max_actual": 0.01},
        "intensity": {"min": -1, "max": 1, "min_actual": 0, "max_actual": 100_000},
        "osi": {"min": -1, "max": 1, "min_actual": -100, "max_actual": 100},
        "order_book_imbalance": {
            "min": -1,
            "max": 1,
            "min_actual": -1,
            "max_actual": 1,
        },
        "current_second": {"min": -1, "max": 1, "min_actual": 0, "max_actual": 60},
        "current_minute": {"min": -1, "max": 1, "min_actual": 0, "max_actual": 60},
        "current_hour": {"min": -1, "max": 1, "min_actual": 0, "max_actual": 24},
    }


class LinearObservation:
    def __init__(
        self,
        obs_info_dict={
            "volatility": {"min": -1, "max": 1, "min_actual": 0, "max_actual": 50},
        },
        n_env=1,
    ):
        if type(obs_info_dict) == LinearObservationSpaces:
            obs_info_dict = obs_info_dict.value
        lows = []
        highs = []

        self.func_dict = {}
        for k, v in obs_info_dict.items():
            for i in ["min", "max", "min_actual", "max_actual"]:
                if i not in v:
                    raise ValueError(f"Missing {i} in obs_info_dict for {k}")
            lows.append(v["min"])
            highs.append(v["max"])
            slope = (v["max_actual"] - v["min_actual"]) / (v["max"] - v["min"])
            intercept = v["min_actual"] - slope * v["min"]

            self.func_dict[k] = {
                "intercept": intercept,
                "slope": slope,
                "min": v["min"],
                "max": v["max"],
                "min_actual": np.full(n_env, v["min_actual"]),
                "max_actual": np.full(n_env, v["max_actual"]),
            }

        self.obs_space = spaces.Box(
            low=np.float32(np.array(lows)),
            high=np.float32(np.array(highs)),
            dtype=np.float32,
        )

    def convert_to_readable(self, obs_dict={"volatility": 50}):
        # convert from normalized to actual
        actual_obs = {}
        for k, v in obs_dict.items():
            calculated_value = (
                self.func_dict[k]["slope"] * v + self.func_dict[k]["intercept"]
            )

            actual_obs[k] = np.minimum(
                np.maximum(calculated_value, self.func_dict[k]["min_actual"]),
                self.func_dict[k]["max_actual"],
            )
        return actual_obs

    def convert_to_normalized(self, obs_dict={"volatility": 0.2}):
        # convert from actual to normalized
        normalized_obs = {}
        for obs_type, value in obs_dict.items():
            calculated_value = (
                value - self.func_dict[obs_type]["intercept"]
            ) / self.func_dict[obs_type]["slope"]
            normalized_obs[obs_type] = np.minimum(
                np.maximum(calculated_value, self.func_dict[obs_type]["min"]),
                self.func_dict[obs_type]["max"],
            )
        obs_list = []
        for k, v in normalized_obs.items():
            obs_list.append(v)
        return np.concatenate(obs_list).T
