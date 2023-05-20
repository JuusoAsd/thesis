
from dotenv import load_dotenv
from src.util import get_test_config
from src.cloning import load_model_by_config
from src.environments.util import setup_venv_config


load_dotenv(".env")
config = get_test_config("test_util")
def test_load_model():
    venv = setup_venv_config(config.data, config.env, config.venv)
    model = load_model_by_config(config, venv=None)
