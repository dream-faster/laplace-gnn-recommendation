from run_pipeline import run_pipeline
from config import Config, link_pred_config

config = link_pred_config
config.epochs = 4
config.k = 12
config.eval_every = 4
config.evaluate_break_at = 50
config.wandb_enabled = True

run_pipeline(config)