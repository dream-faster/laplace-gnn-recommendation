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
    train_data: Union[HeteroData, Data],
    model: Module,
    optimizer: Optimizer,
    config: Config,
) -> tuple[float, Module]:
    criterion = torch.nn.BCEWithLogitsLoss()

    z = model.encode(train_data.x, train_data.edge_index)
    out = model.decode(z, train_data.edge_label_index).view(-1)
    loss = criterion(out, train_data.edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data: Union[HeteroData, Data], model: Module):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()

    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


def epoch_with_dataloader(
    model: Module,
    optimizer: Optimizer,
    train_loader,
    val_loader,
    test_loader,
    config: Config,
):
    for data in iter(train_loader):
        loss = train(data, model, optimizer, config)
    for data in iter(train_loader):
        train_rmse = test(data, model)
    for data in iter(val_loader):
        val_rmse = test(data, model)
    for data in iter(test_loader):
        test_rmse = test(data, model)

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
    train_rmse, model = test(train_loader, model)
    val_rmse, model = test(val_loader, model)
    test_rmse, model = test(test_loader, model)

    return loss, train_rmse, val_rmse, test_rmse
