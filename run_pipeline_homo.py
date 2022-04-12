from regex import E
from tqdm import tqdm
import math
from typing import Union, Tuple

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor

from torch_geometric import seed_everything
from torch_geometric.data import HeteroData, Data


from config import config, Config
from model.encoder_decoder_hetero import Encoder_Decoder_Model_Hetero
from model.encoder_decoder_homo import Encoder_Decoder_Model_Homo
from utils.loss_functions import weighted_mse_loss
from utils.get_info import get_feature_info
from data.types import DataLoaderConfig, FeatureInfo, PipelineConst
from data.data_loader_homo import create_dataloaders_homo, create_datasets_homo
from data.data_loader_hetero import create_dataloaders_hetero, create_datasets_hetero


def select_properties(
    data: Union[HeteroData, Data], config: Config
) -> Union[tuple[dict, dict, dict, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]:
    if config.type == PipelineConst.heterogenous:
        return (
            data.x_dict,
            data.edge_index_dict,
            data["customer", "article"].edge_label_index,
        ), data["customer", "article"].edge_label.float()
    else:  # config.type == PipelineConst.homogenous:
        return (
            data.x,
            data.edge_index,
            data.edge_label_index,
            data.edge_label,
        )


def train(
    data: Union[HeteroData, Data], model: Module, optimizer: Optimizer, config: Config
) -> tuple[float, Module]:
    model.train()
    optimizer.zero_grad()

    x, edge_index, edge_label_index, edge_label = select_properties(data, config)

    pred: torch.Tensor = model(
        x,
        edge_index,
        edge_label_index,
    )
    target = edge_label
    loss: torch.Tensor = weighted_mse_loss(pred, target, None)
    loss.backward()
    optimizer.step()
    return float(loss), model


@torch.no_grad()
def test(data: Union[HeteroData, Data], model: Module, config: Config):
    model.eval()
    x, edge_index, edge_label_index, edge_label = select_properties(data, config)
    pred = model(x, edge_index, edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = edge_label
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse), model


def run_pipeline(config: Config):
    print("| Seeding everything...")
    seed_everything(5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("| Creating Datasets...")
    if config.type == PipelineConst.homogenous:
        loader = create_datasets_homo
    else:
        loader = create_datasets_hetero

    (
        train_loader,
        val_loader,
        test_loader,
        customer_id_map,
        article_id_map,
        full_data,
    ) = loader(DataLoaderConfig(test_split=0.15, val_split=0.15, batch_size=32))
    assert torch.max(train_loader.edge_stores[0].edge_index) <= train_loader.num_nodes

    print("| Creating Model...")
    feature_info = get_feature_info(full_data, config.type)
    if config.type == PipelineConst.heterogenous:
        model = Encoder_Decoder_Model_Hetero(
            hidden_channels=32,
            feature_info=feature_info,
            metadata=train_loader.metadata(),
            embedding=True,
        ).to(device)
    else:
        model = Encoder_Decoder_Model_Homo(
            hidden_channels=32,
            feature_info=feature_info,
            embedding=False,
        ).to(device)

    # Due to lazy initialization, we need to run one model step so the number
    # of parameters can be inferred:
    print("| Lazy Initialization of Model...")
    with torch.no_grad():
        model.initialize_encoder_input_size(train_loader)

    print("| Defining Optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("| Training Model...")
    num_epochs = config.epochs
    loop_obj = tqdm(range(0, num_epochs))
    for epoch in loop_obj:
        loss, model = train(train_loader, model, optimizer, config)
        train_rmse, model = test(train_loader, model, config)
        val_rmse, model = test(val_loader, model, config)
        test_rmse, model = test(test_loader, model, config)

        loop_obj.set_postfix_str(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, "
            f"Val: {val_rmse:.4f}, Test: {test_rmse:.4f}"
        )
        if epoch % math.floor(num_epochs / 3) == 0:
            torch.save(model.state_dict(), f"model/saved/model_{epoch:03d}.pt")
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, "
                f"Val: {val_rmse:.4f}, Test: {test_rmse:.4f}"
            )


if __name__ == "__main__":
    run_pipeline(config)
