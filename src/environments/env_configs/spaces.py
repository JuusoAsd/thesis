from enum import Enum
from copy import deepcopy

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


def create_space(min, max, min_actual, max_actual, external: bool):
    """
    creates a dictionary that determines linear normalization for a given variable
    external means that variable is read from data and then normalized
    not external is something that is calculated from other variables
    """
    return {
        "min": min,
        "max": max,
        "min_actual": min_actual,
        "max_actual": max_actual,
        "external": external,
    }


class LinearObservationSpaces(Enum):
    OnlyInventorySpace = {"inventory": create_space(-1, 1, -2, 2, False)}

    SimpleLinearSpace = {
        "inventory": create_space(-1, 1, -2, 2, False),
        "volatility": create_space(-1, 1, 0, 0.01, True),
        "intensity": create_space(-1, 1, 0, 100_000, True),
    }

    OSILinearSpace = {
        **SimpleLinearSpace,
        "osi": create_space(-1, 1, -100, 100, True),
    }

    EverythingLinearSpace = {
        **OSILinearSpace,
        "order_book_imbalance": create_space(-1, 1, -1, 1, True),
        "current_second": create_space(-1, 1, 0, 60, True),
        "current_minute": create_space(-1, 1, 0, 60, True),
        "current_hour": create_space(-1, 1, 0, 24, True),
    }

    EverythingLinearSpaceAS = {
        "as_bid": create_space(-1, 1, -1, 1, False),
        "as_ask": create_space(-1, 1, -1, 1, False),
        **EverythingLinearSpace,
    }


class LinearObservation:
    """
    numpy version of previous linear observation space
    NOTE: be very very careful with ordering of items in input arrays
    """

    def __init__(self, obs_info_dict, n_env=1):
        if isinstance(obs_info_dict, LinearObservationSpaces):
            self.space_type = obs_info_dict
            obs_info_dict = obs_info_dict.value
        else:
            self.space_type = None
        self.obs_info_dict = deepcopy(obs_info_dict)
        lows = []
        highs = []

        self.func_dict = {}
        self.external = []
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
            if v["external"]:
                self.external.append(k)
        self.slopes = np.array([self.func_dict[k]["slope"] for k in self.func_dict])
        self.intercepts = np.array(
            [self.func_dict[k]["intercept"] for k in self.func_dict]
        )
        self.mins = np.array([self.func_dict[k]["min"] for k in self.func_dict])
        self.maxs = np.array([self.func_dict[k]["max"] for k in self.func_dict])
        self.mins_actual = np.array(
            [self.func_dict[k]["min_actual"] for k in self.func_dict]
        )
        self.maxs_actual = np.array(
            [self.func_dict[k]["max_actual"] for k in self.func_dict]
        )

        self.obs_space = spaces.Box(
            low=np.float32(np.array(lows)),
            high=np.float32(np.array(highs)),
            dtype=np.float32,
        )

    def convert_to_normalized(self, obs_array):
        # convert from actual to normalized
        calculated_values = (obs_array - self.intercepts) / self.slopes
        normalized_obs = np.minimum(np.maximum(calculated_values, self.mins), self.maxs)
        return normalized_obs

    def convert_to_readable(self, obs_array):
        # convert from normalized to actual
        calculated_values = self.slopes * obs_array + self.intercepts
        return calculated_values

    def get_correct_order(self):
        return list(self.func_dict.keys())

    # def get_grid_space(self, n, constant_values={}):
    #     """
    #     returns a grid of observations from min to max for each variable while holding all others constant

    #     n: number of points to sample for each variable
    #     constant: list of what each variable should be set while constant (defaults to middle of the range)
    #     """

    #     obs_values = []
    #     for k, v in self.obs_info_dict.items():
    #         min_val = v["min"]
    #         max_val = v["max"]
    #         distance = max_val - min_val
    #         step = distance / (n)
    #         values = [i for i in np.arange(min_val, max_val + 1e-6, step)]
    #         obs_values.append(values)
    #     grids = np.meshgrid(*obs_values, indexing="ij")
    #     grid_space = np.stack(grids, axis=-1).reshape(-1, len(self.obs_info_dict))

    #     return grid_space

    def get_grid_space(self, n, constant_values={}):
        """
        Returns a grid of observations from min to max for each variable while holding all others constant

        n: number of points to sample for each variable
        constant_values: dict where key is the variable name to be held constant, and value is the constant value
        """

        obs_values = []
        variable_order = {}  # To track the order of variables

        # Create list of obs_values for non-constant variables
        steps = 0
        for k, v in self.obs_info_dict.items():
            variable_order[k] = steps
            if k not in constant_values:
                min_val = v["min"]
                max_val = v["max"]
                distance = max_val - min_val
                step = distance / (n)
                values = [i for i in np.arange(min_val, max_val + 1e-6, step)]
                obs_values.append(values)
                steps += 1

        grids = np.meshgrid(*obs_values, indexing="ij")
        grid_space = np.stack(grids, axis=-1).reshape(
            -1, len(self.obs_info_dict) - len(constant_values)
        )
        # Insert columns for constant variables
        for var, const_val in constant_values.items():
            grid_space = np.insert(grid_space, variable_order[var], const_val, axis=1)

        return np.round(grid_space, 5)
