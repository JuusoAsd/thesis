import logging
from abc import ABC, abstractmethod
import numpy as np


class BaseRewardClass(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def start_step(self):
        """
        Function that is called immediately following step after action is applied
        """
        pass

    @abstractmethod
    def end_step(self):
        """
        Function that is called when all step-related, non-view only operations are complete
        """
        pass


class PnLReward(BaseRewardClass):
    def __init__(self, env, inventory_threshold=0.8, high_penalty=100):
        super().__init__(env)
        self.inventory_threshold = inventory_threshold
        self.high_penalty = high_penalty

    def start_step(self):
        self.value_start = self.env._get_value()

    def end_step(self):
        self.value_end = self.env._get_value()
        profit = self.value_end - self.value_start
        inventory_is_high = np.abs(self.env.norm_inventory) > self.inventory_threshold

        profit -= inventory_is_high * self.high_penalty
        return profit


class InventoryReward(BaseRewardClass):
    def __init__(self, env):
        super().__init__(env)

    def start_step(self):
        pass

    def end_step(self):
        return -np.abs(self.env.norm_inventory)


class SpreadPnlReward(BaseRewardClass):
    """
    Reward based on how much PnL earned due to spread
    """

    def __init__(self, env, inventory_threshold=0.8, high_penalty=100):
        super().__init__(env)
        self.inventory_threshold = inventory_threshold
        self.high_penalty = high_penalty

    def start_step(self):
        pass

    def end_step(self):
        spread = self.env.spread.reshape(-1, 1)
        inventory = np.abs(self.env.norm_inventory)
        inventory_is_high = np.abs(self.env.norm_inventory) > self.inventory_threshold
        return spread / (1 + inventory) - inventory_is_high * self.high_penalty


class MultistepPnl(BaseRewardClass):
    def __init__(self, env, inventory_threshold=0.8, high_penalty=100, steps=10):
        super().__init__(env)
        self.inventory_threshold = inventory_threshold
        self.high_penalty = high_penalty
        self.steps = steps

    def start_step(self):
        pass

    def end_step(self):
        # env keeps track of end values in self.values
        last_value = self.env.values[-1]
        try:
            start_value = self.env.values[-self.steps]
        except IndexError:
            start_value = self.env.values[0]
        inventory_is_high = np.abs(self.env.norm_inventory) > self.inventory_threshold
        return (last_value - start_value) - inventory_is_high * self.high_penalty


class AssymetricPnLDampening(BaseRewardClass):
    """
    Reward is
    - profit / inventory factor if profit is positive
    - profit if profit is negative

    also possible to constantly penalize inventory
    """

    def __init__(
        self,
        env,
        inventory_threshold=0.8,
        high_penalty=10,
        dampening=3,
        inventory_penalty=0.1,
        add_inventory_penalty=False,
    ):
        super().__init__(env)
        self.high_penalty = high_penalty
        self.dampening = dampening
        self.inventory_penalty = inventory_penalty
        self.inventory_threshold = inventory_threshold
        self.add_inventory_penalty = add_inventory_penalty

    def start_step(self):
        self.value_start = self.env._get_value()

    def end_step(self):
        self.value_end = self.env._get_value()
        profit = self.value_end - self.value_start
        is_profit = profit > 0
        norm_inventory = np.abs(self.env.norm_inventory)
        reward = is_profit * (
            profit / (1 + norm_inventory**self.dampening) + (1 - is_profit) * profit
        )
        is_liquidated = norm_inventory > self.inventory_threshold
        reward -= is_liquidated * self.high_penalty
        if self.add_inventory_penalty:
            reward -= norm_inventory * self.inventory_penalty

        return reward


class InventoryIntegralPenalty(BaseRewardClass):
    """
    reward function that
        - rewards for return over last timesteps like multistep_pnl
        - penalizes for both inventory and inventory over time
    """

    def __init__(
        self,
        env,
        inventory_threshold=0.8,
        high_penalty=10,
        steps=10,
        penalty_limit=0.3,  # after which inventory level penalty start accumulating
        over_time_modifier=2,  # how much to penalize inventory over time
        spot_modifier=2,  # how much to penalize immediate inventory
    ):
        super().__init__(env)

        self.inventory_threshold = inventory_threshold
        self.high_penalty = high_penalty
        self.steps = steps
        self.penalty_limit = penalty_limit
        self.over_time_modifier = over_time_modifier
        self.spot_modifier = spot_modifier
        # check that env has n_envs attribute
        if not hasattr(self.env, "n_envs"):
            raise AttributeError("env must have n_envs attribute")
        self.accumulated_inventory = np.full((self.env.n_envs, 1), 0.0)

    def start_step(self):
        pass

    def end_step(self):
        # Increase penalty for inventory over time

        inventory = self.env.norm_inventory
        penalize = inventory > self.penalty_limit

        # MultistepPnL seems to be performing better than PnL, use it for PnL
        last_value = self.env.values[-1]
        try:
            start_value = self.env.values[-self.steps]
        except IndexError:
            start_value = self.env.values[0]

        returns = last_value / start_value - 1
        is_positive = returns > 0

        # if inventory is over limit, accumulate inventory otherwise reset accumulated inventory
        self.accumulated_inventory = (inventory + self.accumulated_inventory) * penalize

        # Penalty for inventory over time should ramp up but not include current inventory so no double count
        accumulation_penalty = (
            self.accumulated_inventory - inventory * penalize
        ) ** self.over_time_modifier

        # spot penalty should also be non-linear
        spot_penalty = np.abs(inventory) ** self.spot_modifier

        # include extra penalty for above threshold
        is_liquidated = np.abs(inventory) > self.inventory_threshold

        # always above 1
        inventory_penalty = (
            1
            + (accumulation_penalty * spot_penalty)
            + is_liquidated * self.high_penalty
        )
        # negative profit -> inventory multiplies reward
        reward = is_positive * (returns / inventory_penalty) + (1 - is_positive) * (
            returns * inventory_penalty
        )
        # logging.debug(
        #     f"inventory: {self.env.norm_inventory}, {self.env.norm_inventory.shape}"
        # )
        # logging.debug(f"returns: {returns}, {returns.shape}")
        # logging.debug(f"accumulated_inventory: {self.accumulated_inventory}")
        # logging.debug(f"reward: {reward}, {reward.shape}")
        return reward


class SimpleInventoryPnlReward(BaseRewardClass):
    def __init__(self, env, inventory_threshold=0.8, high_penalty=100):
        super().__init__(env)
        self.inventory_threshold = inventory_threshold
        self.high_penalty = high_penalty

    def start_step(self):
        self.value_start = self.env._get_value()

    def end_step(self):
        self.value_end = self.env._get_value()
        profit = self.value_end - self.value_start
        inventory = np.abs(self.env.norm_inventory)
        is_high = inventory > self.inventory_threshold

        return profit / (1 + inventory) - is_high * self.high_penalty


reward_dict = {
    "pnl": PnLReward,
    "inventory": InventoryReward,
    "spreadpnl": SpreadPnlReward,
    "multistep_pnl": MultistepPnl,
    "assymetric_pnl_dampening": AssymetricPnLDampening,
    "inventory_integral_penalty": InventoryIntegralPenalty,
    "simple_inventory_pnl_reward": SimpleInventoryPnlReward,
}
