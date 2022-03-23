from utils.types import DataLoaderConfig
from data.data_loader import run_dataloader

config = DataLoaderConfig(test_split=0.15, val_split=0.15)
train, val, test = run_dataloader(config)
