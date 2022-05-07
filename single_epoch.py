from config import Config
from typing import Union, Tuple
import numpy as np

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.data import HeteroData, Data
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from utils.metrics_encoder_decoder import get_metrics_universal
from utils.get_info import select_properties

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    train_data: Union[HeteroData, Data],
    model: Module,
    optimizer: Optimizer,
) -> Tensor:

    x, edge_index, edge_label_index, edge_label = select_properties(train_data)
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()

    out = model(x, edge_index, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(
    data: Union[HeteroData, Data], model: Module, exclude_edge_indices: list, k: int
) -> Tuple[float, float]:

    x, edge_index_dict, edge_label_index, edge_label = select_properties(data)
    output = model.infer(x, edge_index_dict, edge_label_index)

    recall, precision, ndcg = get_metrics_universal(
        output, edge_index_dict, exclude_edge_indices, k=k
    )

    # roc_auc_score = roc_auc_score(edge_label.cpu().numpy(), output.cpu().numpy())

    return recall, precision


def epoch_with_dataloader(
    model: Module,
    optimizer: Optimizer,
    train_loader: Union[LinkNeighborLoader, NeighborLoader],
    val_loader,
    test_loader,
    epoch_id: int,
    config: Config,
):
    losses, val_recalls, val_precisions = [], [], []

    train_loop = tqdm(iter(train_loader), colour="blue")
    for i, data in enumerate(train_loop):
        train_loop.set_description(f"TRAIN | epoch: {epoch_id}")
        loss = train(data.to(device), model, optimizer)
        losses.append(loss.detach().cpu().item())
        train_loop.set_postfix_str(f"Loss: {np.mean(losses):.4f}")

        if config.profiler is not None:
            config.profiler.print_stats(i)

    val_loop = tqdm(iter(val_loader), colour="yellow")
    for i, data in enumerate(val_loop):
        if config.evaluate_break_at and i == config.evaluate_break_at:
            break
        val_loop.set_description(f"VAL | epoch: {epoch_id}")
        val_recall, val_precision = test(data.to(device), model, [], k=config.k)
        val_recalls.append(val_recall)
        val_precisions.append(val_precision)
        val_loop.set_postfix_str(
            f"Recall: {np.mean(val_recalls):.4f} | Precision: {np.mean(val_precisions):.4f}"
        )

    return np.mean(val_precisions)
