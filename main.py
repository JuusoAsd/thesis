import logging
from src.environments.mm_env import MMEnv
from src.environments.as_agent import ASAgent
from csv_parser.AS.parse_as import parse_as_full
import time

if __name__ == "__main__":
    target = r"C:\Users\Ville\Documents\gradu\parsed_data\AS\data_full.csv"
    agent_params = {"risk_aversion": 0.5}

    # get current timestamp
    ts = time.time()

    logging.basicConfig(
        filename=f"logs/as_env_sample_{int(ts)}.log",
        encoding="utf-8",
        level=logging.INFO,
    )
    env = MMEnv(
        target,
        ASAgent,
        agent_parameters=agent_params,
        price_decimals=4,
        step_aggregation=1_000,
        logging=True,
        logger=logging.getLogger(__name__),
    )
    env.reset()
    # for i in range(1000):
    #     env.step(None)
    while True:
        try:
            done = env.step(None)
            if done:
                break
        except Exception as e:
            print(e)
            break

    print(
        f"Cash: {env.quote_asset}, inventory: {env.base_asset}, value: {env.get_current_value()}"
    )
    exit()
