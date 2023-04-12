from stable_baselines3.common.vec_env import VecNormalize
from environments.mm_env_vec import MMVecEnv, SBMMVecEnv
from environments.env_configs.rewards import *
from environments.env_configs.spaces import (
    ActionSpace,
    LinearObservationSpaces,
    LinearObservation,
)


def setup_venv(
    data,
    obs_space=LinearObservation(LinearObservationSpaces.SimpleLinearSpace),
    act_space=ActionSpace.NoSizeAction,
    reward_class=InventoryIntegralPenalty,
    n_env=1,
    normalize=False,
    env_params={},
    venv_params={},
):
    column_mapping = {col: n for (n, col) in enumerate(data.columns)}
    env = MMVecEnv(
        data.to_numpy(),
        n_envs=n_env,
        params={
            "observation_space": obs_space,
            "action_space": act_space,
        },
        column_mapping=column_mapping,
        reward_class=reward_class,
        **env_params,
    )
    venv = SBMMVecEnv(env, **venv_params)

    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=100_000)
    return venv


# class FileManager:
#     def __init__(self, folder_path, output_type=None, headers=True):
#         self.path = folder_path
#         self.headers = headers
#         self.create_output = output_type
#         if os.path.isdir(folder_path):
#             dir_files = os.listdir(self.path)
#             dir_files.sort()
#             self.files = []
#             for i in dir_files:
#                 self.files.append(os.path.join(self.path, i))
#         else:
#             self.files = [folder_path]
#         ok = self.get_next_file()
#         if ok is False:
#             raise Exception("No files in folder")

#     def get_next_file(self):
#         if len(self.files) > 0:
#             self.iterator = iter(open(self.files.pop(0)))
#             if self.headers:
#                 self.iterator.__next__()
#             return True
#         else:
#             return False

#     def get_next_event(self, start=False):
#         # try get the next line from current file
#         try:
#             val = next(self.iterator)
#         except StopIteration as e:
#             # print(e)
#             continues = self.get_next_file()
#             if not continues:
#                 return None
#             val = next(self.iterator)
#         if self.create_output is None:
#             return val.rstrip().split(",")
#         else:
#             return self.create_output(val.rstrip().split(","))


# class Trade:
#     def __init__(self, price, size):
#         self.price = price
#         self.size = size


# class CurrentState:
#     """
#     ts, mid_price, 25 best bids and asks with sizes
#     50 is best bid, 52 is best ask
#     """

#     def __init__(self, input_list):
#         self.timestamp = int(input_list[0])
#         self.best_bid = float(input_list[50])
#         self.best_ask = float(input_list[52])

#     def set_trades(self, trades):
#         self.trades = trades


# class StateManager:
#     def __init__(self, trade_folder, orderbook_folder):
#         self.trades_manager = FileManager(trade_folder, Trade)
#         self.orderbook_manager = FileManager(orderbook_folder, CurrentState)
#         self.initialized = False

#     def get_next_state(self):
#         if not self.initialized:
#             self.initialized = True
#             return self.initialize()

#     def initialize(self):
#         """Want to make sure that the first trade is JUST before the first orderbook"""
#         current_ob = self.orderbook_manager.get_next_event()
#         current_trade = self.trades_manager.get_next_event()


# class ASState:
#     def __init__(self, input_list):
#         # bid, ask, trade price, trade size, vol, intensity
#         self.timestamp = float(input_list[0])
#         self.best_bid = float(input_list[1])
#         self.best_ask = float(input_list[2])
#         self.mid_price = (self.best_bid + self.best_ask) / 2
#         self.trade_price = float(input_list[3])
#         self.trade_size = float(input_list[4])
#         self.volatility = float(input_list[5])
#         self.intensity = float(input_list[6])

#     def get_observation(self):
#         return np.array([self.best_bid, self.best_ask, self.vol, self.intensity])
