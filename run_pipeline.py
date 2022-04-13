from tqdm import tqdm
import math
from typing import Callable, Union, Tuple

import torch

from torch_geometric import seed_everything
from torch_geometric.data import HeteroData, Data


from config import config, Config
from model.encoder_decoder_hetero import Encoder_Decoder_Model_Hetero
from model.encoder_decoder_homo import Encoder_Decoder_Model_Homo

from utils.get_info import get_feature_info
from data.types import DataLoaderConfig, FeatureInfo, GraphType
from data.data_loader_homo import create_dataloaders_homo, create_datasets_homo
from data.data_loader_hetero import create_dataloaders_hetero, create_datasets_hetero

from single_epoch import epoch_with_dataloader, epoch_without_dataloader


def select_loader_epochloop(config: Config) -> tuple[Callable, Callable]:
    if config.type == GraphType.homogenous:
        if config.dataloader:
            return create_dataloaders_homo, epoch_with_dataloader
        else:
            return create_datasets_homo, epoch_without_dataloader
    else:
        if config.dataloader:
            return create_dataloaders_hetero, epoch_with_dataloader
        else:
            return create_datasets_hetero, epoch_without_dataloader


def run_pipeline(config: Config):
    print(f"--- Pipeline Type: {config.type} ---")
    print("| Seeding everything...")
    seed_everything(5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("| Creating Datasets...")
    loader, epoch_loop = select_loader_epochloop(config)
    (
        train_loader,
        val_loader,
        test_loader,
        customer_id_map,
        article_id_map,
        full_data,
    ) = loader(DataLoaderConfig(test_split=0.001, val_split=0.001, batch_size=32))

    print(
        "--- Data Type: {} ---".format(
            GraphType.heterogenous
            if type(full_data) == HeteroData
            else GraphType.homogenous
        )
    )
    print("| Creating Model...")
    feature_info = get_feature_info(full_data, config.type)
    if config.type == GraphType.heterogenous:
        model = Encoder_Decoder_Model_Hetero(
            hidden_channels=32,
            feature_info=feature_info,
            metadata=next(iter(train_loader)).metadata(),
            embedding=False,
        ).to(device)
    else:
        model = Encoder_Decoder_Model_Homo(
            in_channels=8,
            out_channels=1,
            hidden_channels=57,
        ).to(device)

    # Due to lazy initialization, we need to run one model step so the number
    # of parameters can be inferred:
    print("| Lazy Initialization of Model...")
    with torch.no_grad():
        if config.dataloader:
            model.initialize_encoder_input_size(next(iter(train_loader)))
        else:
            model.initialize_encoder_input_size(train_loader)

    print("| Defining Optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("| Training Model...")
    loop_obj = tqdm(range(0, config.epochs))
    for epoch in loop_obj:
        loss, train_rmse, val_rmse, test_rmse = epoch_loop(
            model, optimizer, train_loader, val_loader, test_loader, config
        )
        torch.save(model.state_dict(), f"model/saved/model_{epoch:03d}.pt")


if __name__ == "__main__":
    run_pipeline(config)
