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


def select_properties(
    data: Union[HeteroData, Data], config: Config
) -> Union[tuple[dict, dict, dict, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]:
    if config.type == GraphType.heterogenous:
        return (
            data.x_dict,
            data.edge_index_dict,
            data["customer", "article"].edge_label_index,
            data["customer", "article"].edge_label.float(),
        )
    else:  # config.type == GraphType.homogenous:
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

    pred: Tensor = model(
        x,
        edge_index,
        edge_label_index,
    )
    target = edge_label
    loss: Tensor = weighted_mse_loss(pred, target, None)
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


def epoch_with_dataloader(
    model: Module,
    optimizer: Optimizer,
    train_loader,
    val_loader,
    test_loader,
    config: Config,
):
    for data in iter(train_loader):
        loss, model = train(data, model, optimizer, config)
    for data in iter(train_loader):
        train_rmse, model = test(data, model, config)
    for data in iter(val_loader):
        val_rmse, model = test(data, model, config)
    for data in iter(test_loader):
        test_rmse, model = test(data, model, config)

    return loss, train_rmse, val_rmse, test_rmse


def epoch_without_dataloader(
    model: Module,
    optimizer: Optimizer,
    train_loader: Union[HeteroData, Data],
    val_loader: Union[HeteroData, Data],
    test_loader: Union[HeteroData, Data],
    config: Config,
):
    loss, model = train(train_loader, model, optimizer, config)
    train_rmse, model = test(train_loader, model, config)
    val_rmse, model = test(val_loader, model, config)
    test_rmse, model = test(test_loader, model, config)

    return loss, train_rmse, val_rmse, test_rmse
