from config import Config
from typing import Optional
from reporting.types import Stats

import wandb
import os


def get_wandb():
    from dotenv import load_dotenv

    load_dotenv()

    """ 0. Login to Weights and Biases """
    wsb_token = os.environ.get("WANDB_API_KEY")
    if wsb_token:
        wandb.login(key=wsb_token)
        return wandb
    else:
        return None  # wandb.login()


def launch_wandb(project_name: str, default_config: Config) -> Optional[object]:
    wandb = get_wandb()
    if wandb is None:
        raise Exception(
            "Wandb can not be initalized, the environment variable WANDB_API_KEY is missing (can also use .env file)"
        )
    else:
        wandb.init(project=project_name, config=vars(default_config), reinit=True)
        return wandb


def override_config_with_wandb_values(
    wandb: Optional[object], raw_config: Config
) -> Config:
    if wandb is None:
        return raw_config

    wandb_config: dict = wandb.config

    config_dict = vars(raw_config)
    for k in config_dict:
        config_dict[k] = wandb_config[k]

    return Config(**config_dict)


def send_report_to_wandb(stats: Stats, wandb: Optional[object]):
    if wandb is None:
        return

    run = wandb.run
    run.save()

    for key, value in stats.items():
        run.log({key: value})

    run.finish()


def setup_config(
    project_name: str, with_wandb: bool, raw_config: Config
) -> tuple[Optional[object], Config]:
    wandb = None
    config = raw_config
    if with_wandb:
        wandb = launch_wandb(project_name=project_name, default_config=raw_config)
        new_config = override_config_with_wandb_values(wandb, raw_config)
        config = new_config

    return wandb, config
