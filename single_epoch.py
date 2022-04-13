import math
from config import Config
from typing import Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from utils.loss_functions import weighted_mse_loss
from torch_geometric.data import HeteroData, Data
from data.types import GraphType
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score


def select_properties(
    data: Union[HeteroData, Data], config: Config
) -> Union[tuple[dict, dict, dict, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]:
    if config.type == GraphType.heterogenous:
        return (
            data.x_dict,
            data.edge_index_dict,
            data[("customer", "buys", "article")].edge_label_index,
            data[("customer", "buys", "article")].edge_label.float(),
        )
    else:  # config.type == GraphType.homogenous:
        return (
            data.x,
            data.edge_index,
            data.edge_label_index,
            data.edge_label,
        )


def train(
    train_data: Union[HeteroData, Data],
    model: Module,
    optimizer: Optimizer,
    config: Config,
) -> tuple[float, Module]:

    x, edge_index, edge_label_index, edge_label = select_properties(train_data, config)
    criterion = torch.nn.BCEWithLogitsLoss()

    z = model.encoder(x, edge_index)
    out = model.decoder(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data: Union[HeteroData, Data], model: Module, config: Config) -> float:
    x, edge_index, edge_label_index, edge_label = select_properties(data, config)

    model.eval()
    z = model.encoder(x, edge_index)
    out = model.decoder(z, edge_label_index).view(-1).sigmoid()

    return roc_auc_score(edge_label.cpu().numpy(), out.cpu().numpy())


def epoch_with_dataloader(
    model: Module,
    optimizer: Optimizer,
    train_loader,
    val_loader,
    test_loader,
    config: Config,
):
    loop_obj = tqdm(iter(train_loader))
    for data in loop_obj:
        loss = train(data, model, optimizer, config)
        loop_obj.set_postfix_str(f"Loss: {loss:.4f}")
    # for data in iter(train_loader): there's no way we can loop through the train dataset again, one epoch takes ages
    #     train_rmse = test(data, model, config)
    # for data in iter(val_loader):
    #     val_rmse = test(data, model, config)
    # for data in iter(test_loader):
    #     test_rmse = test(data, model, config)

    val_rmse = test(val_loader, model, config)
    test_rmse = test(test_loader, model, config)

    return loss, val_rmse, test_rmse


def epoch_without_dataloader(
    model: Module,
    optimizer: Optimizer,
    train_loader: Union[HeteroData, Data],
    val_loader: Union[HeteroData, Data],
    test_loader: Union[HeteroData, Data],
    config: Config,
):
    loss, model = train(train_loader, model, optimizer, config)
    train_rmse, model = test(train_loader, model)
    val_rmse, model = test(val_loader, model)
    test_rmse, model = test(test_loader, model)

    return loss, train_rmse, val_rmse, test_rmse
