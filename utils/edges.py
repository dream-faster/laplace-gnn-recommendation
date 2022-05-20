import torch as t
from torch import Tensor
from typing import Optional, List
from collections import defaultdict


def create_adj_dict(edge_index: Tensor, from_nodes: Optional[Tensor] = None) -> dict:
    """Generates dictionary of items for each user

    Args:
        edge_index (t.Tensor): 2 by N list of edges

    Returns:
        dict: dictionary of items for each user
    """
    from_nodes = set(from_nodes.tolist()) if from_nodes is not None else None
    sorted_edges = t.sort(edge_index, dim=1).values
    items_per_user: dict = defaultdict(lambda: t.tensor([], dtype=t.long))
    for edge in sorted_edges:
        from_user = edge[0].item()
        if from_nodes is not None and from_user not in from_nodes:
            continue
        items_per_user[from_user] = t.cat(
            (items_per_user[from_user], edge[1].unsqueeze(dim=0)), dim=0
        )
    return items_per_user


def create_adj_list(
    edge_index: Tensor, from_nodes: Optional[Tensor] = None
) -> List[Tensor]:
    """Generates list of items for each user

    Args:
        edge_index (t.Tensor): 2 by N list of edges

    Returns:
        tensor: List[Tensor] of items for each user
    """
    users = edge_index[0].unique(sorted=True) if from_nodes is None else from_nodes
    return [edge_index[1][edge_index[0] == user] for user in users]

    sorted_edges = t.sort(edge_index, dim=1).values
    items_per_user: dict = defaultdict(lambda: t.tensor([], dtype=t.long))
    for edge in sorted_edges:
        items_per_user[edge[0].item()] = t.cat(
            (items_per_user[edge[0].item()], edge[1].unsqueeze(dim=0)), dim=0
        )
    return items_per_user
