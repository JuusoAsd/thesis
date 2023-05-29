import os
import sys

sys.path.append(f"{os.getcwd()}/gradu")

from omegaconf import OmegaConf
import time
import os


from dotenv import load_dotenv

from stable_baselines3 import PPO


from src.environments.env_configs.policies import ASPolicyVec
from src.environments.util import setup_venv_config
from src.model_testing import trained_vs_manual
from src.data_management import get_data_by_dates
from src.util import get_config, create_config_hash
from src.cloning import CloneDuration, clone_bc, load_trained_model

load_dotenv()


def create_cloned_models():
    config = get_config("clone_config_multiple")
    config.env.params.as_expert_params = config.expert_params
    venv = setup_venv_config(config.clone_data, config.env, config.venv)
    cloning_duration = CloneDuration.Short
    expert = ASPolicyVec(env=venv.env, **config.expert_params)

    for i in config.policy_kwargs:
        clone_model_config = OmegaConf.create(
            {
                "policy_kwargs": i,
                "action": config.env.spaces.action_space,
                "observation": config.env.spaces.observation_space.params,
            }
        )
        start_time = time.time()
        hash = create_config_hash(clone_model_config)

        config_to_dict = OmegaConf.to_container(i, resolve=True)
        student_model = PPO("MlpPolicy", venv, verbose=1, policy_kwargs=config_to_dict)
        model_name = clone_bc(
            venv,
            expert,
            student_model,
            cloning_duration,
            testing=False,
            model_name=hash,
        )
        model = load_trained_model(hash, venv, False)

        if config.compare:
            single_venv = venv.clone_venv(
                get_data_by_dates(**config.verify_cloning_data).to_numpy()
            )
            metrics_model, metrics_expert = trained_vs_manual(single_venv, model, False)
            print(f"Model: {metrics_model}")
            print(f"Expert: {metrics_expert}")
            print(f"DONE, took {round((time.time() - start_time)/60,2)} minutes\n")
