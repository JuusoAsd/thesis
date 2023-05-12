from datetime import datetime, timedelta
from stable_baselines3 import PPO

from src.data_management import get_data_by_dates
from src.cloning import load_trained_model, save_trained_model
from src.util import get_config
from src.model_testing import test_trained_vs_manual
from src.environments.util import setup_venv
from src.environments.env_configs.spaces import ActionSpace
from src.environments.env_configs.rewards import (
    PnLReward,
    AssymetricPnLDampening,
    InventoryIntegralPenalty,
    SpreadPnlReward,
)
from src.environments.env_configs.callbacks import ExternalMeasureCallback


def train_model():
    load_model = True
    dates = "2021_12_23"
    data = get_data_by_dates(dates)

    venv = setup_venv(
        data=data,
        act_space=ActionSpace.NormalizedAction,
        reward_class=AssymetricPnLDampening,
        inv_envs=5,
        time_envs=4,
        env_params={"inv_jump": 0.18, "data_portion": 0.5},
    )
    # model = PPO("MlpPolicy", venv, verbose=1)
    model = load_trained_model("clone_bc", venv)
    model.learn(total_timesteps=1_000_000)
    save_trained_model(model, "clone_to_assymetricpnl")


def initial_training():
    start_date = datetime.strptime("2021_12_23", "%Y_%m_%d")
    duration = 7
    end_date = start_date + timedelta(days=duration)
    data = get_data_by_dates(start_date, end_date)
    reward = InventoryIntegralPenalty
    print(reward.__name__)
    venv = setup_venv(
        data=data,
        act_space=ActionSpace.NormalizedAction,
        reward_class=reward,
        inv_envs=5,
        time_envs=4,
        env_params={"inv_jump": 0.18, "data_portion": 0.5},
    )
    model = PPO("MlpPolicy", venv, verbose=1)
    model.learn(total_timesteps=5_000_000)
    start_str = start_date.strftime("%Y_%m_%d")
    end_str = end_date.strftime("%Y_%m_%d")
    save_trained_model(model, f"clone_to_{reward.__name__}_{start_str}_to_{end_str}")


def rolling_train_test():
    date = datetime.strptime("2021_12_23", "%Y_%m_%d")

    first = True
    for i in range(15):
        data = get_data_by_dates(date)
        previous_date = date - timedelta(days=1)

        training_venv = setup_venv(
            data=data,
            act_space=ActionSpace.NormalizedAction,
            reward_class=PnLReward,
            inv_envs=5,
            time_envs=4,
            env_params={"inv_jump": 0.18, "data_portion": 0.5},
        )

        if not first:
            test_venv = setup_venv(
                data=data,
                act_space=ActionSpace.NormalizedAction,
                reward_class=PnLReward,
                inv_envs=1,
                time_envs=1,
                env_params={"inv_jump": 0.18, "data_portion": 0.5},
            )
            test_model = load_trained_model(
                f"clone_to_pnlreward_{previous_date.strftime('%Y_%m_%d')}", test_venv
            )
            test_trained_vs_manual(test_venv, test_model)
            model = load_trained_model(
                f"clone_to_pnlreward_{previous_date.strftime('%Y_%m_%d')}",
                training_venv,
            )
        else:
            model = load_trained_model("clone_bc", training_venv)
            first = False
        print(f"-" * 30)
        model.learn(total_timesteps=1_000_000, log_interval=None)
        save_trained_model(model, f"clone_to_pnlreward_{date.strftime('%Y_%m_%d')}")

        date += timedelta(days=1)


def train_with_callback():
    start_date = datetime.strptime("2021_12_23", "%Y_%m_%d")
    duration = 7
    end_date = start_date + timedelta(days=duration)
    data = get_data_by_dates(start_date, end_date)
    reward = InventoryIntegralPenalty
    venv = setup_venv(
        data=data,
        act_space=ActionSpace.NormalizedAction,
        reward_class=reward,
        inv_envs=5,
        time_envs=4,
        env_params={"inv_jump": 0.18, "data_portion": 0.5},
    )
    model = PPO("MlpPolicy", venv, verbose=1)

    # create callback using the data from before start date (used in cloning)
    callback = ExternalMeasureCallback(
        data=get_data_by_dates(start_date - timedelta(days=1)).to_numpy(),
        venv=venv,
        wait=2,
        freq=5,
        patience=5,
        verbose=1,
    )
    model.learn(total_timesteps=10_000_000, callback=callback, log_interval=None)
    start_str = start_date.strftime("%Y_%m_%d")
    end_str = end_date.strftime("%Y_%m_%d")


def test_multiple_rewards():
    reward_list = [
        # InventoryIntegralPenalty,
        # MultistepPnl,
        # AssymetricPnLDampening,
        # PnLReward,
        # InventoryReward,
        # SimpleInventoryPnlReward,
        SpreadPnlReward,
    ]
    param_list = []
    # param_list = [
    #     {
    #         "liquidation_threshold": 0.5,
    #     },
    #     {
    #         "liquidation_threshold": 0.8,
    #     },
    # ]
    start_date = datetime.strptime("2021_12_23", "%Y_%m_%d")
    duration = 7
    end_date = start_date + timedelta(days=duration)
    data = get_data_by_dates(start_date, end_date)

    for reward in reward_list:
        # for params in param_list:
        try:
            print(reward.__name__)
            # print(params)
            venv = setup_venv(
                data=data,
                act_space=ActionSpace.NormalizedAction,
                reward_class=reward,
                inv_envs=5,
                time_envs=4,
                env_params={
                    "inv_jump": 0.18,
                    "data_portion": 0.5,
                    # "reward_params": params,
                },
            )
            model = load_trained_model("clone_bc", venv)
            callback = ExternalMeasureCallback(
                data=get_data_by_dates(start_date - timedelta(days=1)).to_numpy(),
                venv=venv,
                model_name=f"cloned_{reward.__name__}_best",
                wait=2,
                freq=5,
                patience=5,
                verbose=1,
            )
            model.learn(
                total_timesteps=10_000_000, callback=callback, log_interval=None
            )
        except Exception as e:
            print(f"Failed: {reward.__name__}, {e}")
            continue

        print(f"-" * 30)


def train_from_model(venv, model, callback, steps=1_000_000):
    model = load_trained_model(model, venv)
    model.learn(total_timesteps=steps, callback=callback, log_interval=None)


def test_init_rolling_train_early_stop():
    config = get_config("rolling_train_test")
    # start with cloned model, train for n days, do rolling training with early stopping
    start_date = datetime.strptime(config.start_date, "%Y_%m_%d")
    duration = config.initial_train_duration
    end_date = start_date + timedelta(days=duration)
    data = get_data_by_dates(start_date, end_date)
    reward = AssymetricPnLDampening
    reward_params = {"liquidation_threshold": 0.8}
    venv = setup_venv(
        data=data,
        act_space=ActionSpace.NormalizedAction,
        reward_class=reward,
        inv_envs=5,
        time_envs=4,
        env_params={
            "inv_jump": 0.18,
            "data_portion": 0.5,
            "reward_params": reward_params,
        },
    )
    callback_start = datetime.strptime(config.start_date_callback, "%Y_%m_%d")
    callback_end = callback_start + timedelta(days=config.callback_duration)
    callback_data = get_data_by_dates(callback_start, callback_end).to_numpy()
    init_callback = ExternalMeasureCallback(
        data=callback_data,
        venv=venv,
        verbose=1,
        model_name=f"cloned_{reward.__name__}_init",
        save_best_model=True,
        **config.callback_params.init,
    )
    model = load_trained_model("clone_bc", venv, model_kwargs=config.model_params)
    # model.learn(total_timesteps=10_000_000, callback=init_callback, log_interval=None)

    # based on the callback result load best model
    current_date = end_date + timedelta(days=1)
    eval_venv = venv.clone_venv(data=get_data_by_dates(current_date).to_numpy())
    eval_model = load_trained_model(f"cloned_{reward.__name__}_init", eval_venv)

    # compare performance of eval model vs manual
    test_trained_vs_manual(
        eval_venv,
        eval_model,
        save_values=True,
        result_file="rolling_test",
        date=current_date,
    )

    train_venv = eval_venv
    # for next 5 days, do rolling training with early stopping
    # first date is already tested, no train on it
    train_venv = eval_venv.clone_venv(
        data=get_data_by_dates(current_date).to_numpy(),
        time_envs=4,
        inv_envs=5,
        inv_jump=0.18,
        data_portion=0.5,
    )
    train_model = load_trained_model(
        f"cloned_{reward.__name__}_init", train_venv, model_kwargs=config.model_params
    )
    start = True
    while current_date < datetime.strptime(config.rolling_end_date, "%Y_%m_%d"):
        print(f"Training on {current_date}")
        current_data = get_data_by_dates(current_date)
        if abs((current_data.best_bid - current_data.best_ask).mean()) > 0.01:
            print(f"Skipping {current_date}")
            current_date += timedelta(days=1)
            continue
        if not start:
            train_venv = eval_venv.clone_venv(
                data=current_data.to_numpy(),
                time_envs=4,
                inv_envs=5,
                inv_jump=0.18,
                data_portion=0.5,
            )
            train_model = load_trained_model(
                f"cloned_{reward.__name__}_train_eval",
                train_venv,
                model_kwargs=config.model_params,
            )

        # evaluate also on same date
        train_eval = ExternalMeasureCallback(
            data=current_data.to_numpy(),
            venv=train_venv,
            verbose=1,
            model_name=f"cloned_{reward.__name__}_train_eval",
            save_best_model=True,
            **config.callback_params.init,
        )
        train_model.learn(
            total_timesteps=10_000_000, callback=train_eval, log_interval=None
        )

        # evaluate on next day
        current_date += timedelta(days=1)
        eval_venv = venv.clone_venv(data=get_data_by_dates(current_date).to_numpy())
        eval_model = load_trained_model(
            f"cloned_{reward.__name__}_train_eval",
            eval_venv,
            model_kwargs=config.model_params,
        )
        test_trained_vs_manual(
            eval_venv,
            eval_model,
            save_values=True,
            result_file="rolling_test",
            date=current_date,
        )
        start = False


if __name__ == "__main__":
    # train_model()
    # rolling_train_test()
    # initial_training()
    # train_with_callback()
    # test_multiple_rewards()
    test_init_rolling_train_early_stop()
    # test_multiple_rewards()
