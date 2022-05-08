from reporting.wandb import send_report_to_wandb
import pandas as pd
from config import Config
from reporting.types import Stats


def report_results(
    output_stats: Stats,
    config: Config,
    wandb,
):
    send_report_to_wandb(output_stats, wandb)
