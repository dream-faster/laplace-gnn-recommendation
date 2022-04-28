from config import Config
from typing import Union, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.data import HeteroData, Data
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from utils.metrics import get_metrics_universal


def select_properties(
    data: Union[HeteroData, Data]
) -> Tuple[dict, dict, Tensor, Tensor]:

    return (
        data.x_dict,
        data.edge_index_dict,
        data[("customer", "buys", "article")].edge_label_index,
        data[("customer", "buys", "article")].edge_label.float(),
    )


def train(
    train_data: Union[HeteroData, Data],
    model: Module,
    optimizer: Optimizer,
) -> float:

    x, edge_index, edge_label_index, edge_label = select_properties(train_data)
    criterion = torch.nn.BCEWithLogitsLoss()

    out = model(x, edge_index, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


# @torch.no_grad()
# def test(data: Union[HeteroData, Data], model: Module) -> float:
#     x, edge_index, edge_label_index, edge_label = select_properties(data)

#     model.eval()
#     out = model(x, edge_index, edge_label_index).view(-1)

#     return roc_auc_score(edge_label.cpu().numpy(), out.cpu().numpy())


@torch.no_grad()
def test(
    data: Union[HeteroData, Data], model: Module, exclude_edge_indices: list
) -> float:

    x, edge_index_dict, edge_label_index, edge_label = select_properties(data)
    output = model.infer(x, edge_index_dict, edge_label_index)

    recall, precision, ndcg = get_metrics_universal(
        output, edge_index_dict, exclude_edge_indices, k=2
    )

    return recall


def epoch_with_dataloader(
    model: Module,
    optimizer: Optimizer,
    train_loader: Union[LinkNeighborLoader, NeighborLoader],
    val_loader,
    test_loader,
):
    loop_obj = tqdm(iter(train_loader))
    for data in loop_obj:
        loss = train(data, model, optimizer)
        loop_obj.set_postfix_str(f"Loss: {loss:.4f}")
    for data in iter(val_loader):
        val_rmse = test(data, model, [])
    for data in iter(test_loader):
        test_rmse = test(
            data,
            model,
            [],
        )

    # val_rmse = test(val_loader, model)
    # test_rmse = test(test_loader, model)

    return loss, val_rmse, test_rmse
