from re import X
from typing import Union, Tuple
import numpy as np
import torch as t
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.data import HeteroData, Data
from torch_geometric.loader import DataLoader
from tqdm.autonotebook import tqdm
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from utils.metrics_encoder_decoder import get_metrics_universal
from utils.get_info import select_properties
from typing import List, Tuple, Optional
from utils.constants import Constants
from model.encoder_decoder import Encoder_Decoder_Model


def __train(
    train_data: Union[HeteroData, Data],
    model: Module,
    optimizer: Optimizer,
) -> Tensor:

    x, edge_index, edge_label_index, edge_label = select_properties(train_data)
    criterion = t.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()

    out = model(x, edge_index, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@t.no_grad()
def __test(
    data: Union[HeteroData, Data],
    model: Encoder_Decoder_Model,
    exclude_edge_indices: list,
    k: int,
) -> Tuple[float, float]:

    x, edge_index_dict, edge_label_index, edge_label = select_properties(data)
    output = model.infer(x, edge_index_dict, edge_label_index)

    recall, precision, ndcg = get_metrics_universal(
        output,
        edge_index_dict[Constants.edge_key],
        edge_label_index,
        exclude_edge_indices,
        k=k,
    )

    return recall, precision


Losses = List[float]


def train_with_dataloader(
    model: Module,
    optimizer: Optimizer,
    data_loader: Union[LinkNeighborLoader, NeighborLoader, DataLoader],
    epoch: int,
    device: str,
) -> Losses:
    losses = []

    train_loop = tqdm(iter(data_loader), colour="blue")
    for i, data in enumerate(train_loop):
        train_loop.set_description(f"TRAIN | epoch: {epoch}")
        loss = __train(data.to(device), model, optimizer)
        losses.append(loss.detach().cpu().item())
        train_loop.set_postfix_str(f"Loss: {np.mean(losses):.4f}")

    return losses


Recall = float
Precision = float


def test_with_dataloader(
    mode: str,  # "VAL" or "TEST"
    model: Encoder_Decoder_Model,
    data_loader: Union[LinkNeighborLoader, NeighborLoader, DataLoader],
    device: str,
    k: int,
    break_at: Optional[int],
) -> Tuple[Recall, Precision]:
    recalls, precisions = [], []
    loop = tqdm(iter(data_loader), colour="yellow")
    for i, data in enumerate(loop):
        if break_at and i == break_at:
            break
        loop.set_description(f"{mode}")
        recall, precision = __test(data.to(device), model, [], k=k)
        recalls.append(recall)
        precisions.append(precision)
        loop.set_postfix_str(
            f"Recall: {np.mean(recalls):.4f} | Precision: {np.mean(precisions):.4f}"
        )

    return np.mean(recalls), np.mean(precisions)
