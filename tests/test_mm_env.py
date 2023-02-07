import logging
from gym.wrappers.normalize import NormalizeObservation

from src.environments.mm_env import MMEnv
from src.environments.policies import AvellanedaStoikovPolicy
from src.environments.util import ASState
from stable_baselines3.common.env_checker import check_env


logging.basicConfig(
    filename=".logs/as_env_sample.log", encoding="utf-8", level=logging.INFO
)

env = MMEnv(
    state_folder="./parsed_data/AvellanedaStoikov/AS_full.csv",
    policy=AvellanedaStoikovPolicy,
    params={
        "tick_size": 0.0001,
        "capital": 1000,
        "step_interval": 10_000,
        "price_decimals": 4,
        "output_type": ASState,
    },
    policy_params={"risk_aversion": 0.1, "inventory_target": 0, "order_size": 1},
    logging=True,
    logger=logging.getLogger(__name__),
)
# env = NormalizeObservation(env)
print("obs sapce", env.reset())


def test_attr():
    assert env.capital == 1000
    assert env.step_interval == 10_000
    assert env.price_decimals == 4


def test_check_env():
    for i in range(1000):
        obs, _, _, _ = env.step(env.action_space.sample())
        print(obs)
    check_env(env)
