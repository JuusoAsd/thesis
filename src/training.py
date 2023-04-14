from stable_baselines3 import PPO

from environments.util import setup_venv
from data_management import get_data_by_dates
from cloning import load_trained_model, save_trained_model
from environments.env_configs.spaces import ActionSpace
from environments.env_configs.rewards import PnLReward


def train_model():
    load_model = True
    dates = "2021_12_23"
    data = get_data_by_dates(dates)

    venv = setup_venv(
        data=data,
        act_space=ActionSpace.NormalizedAction,
        reward_class=PnLReward,
        inv_envs=5,
        time_envs=4,
        env_params={"inv_jump": 0.18, "data_portion": 0.5},
    )
    # model = PPO("MlpPolicy", venv, verbose=1)
    model = load_trained_model("clone_bc", venv)
    model.learn(total_timesteps=1_000_000)
    save_trained_model(model, dates)


if __name__ == "__main__":
    train_model()
