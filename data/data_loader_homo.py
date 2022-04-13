from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
from data.types import DataLoaderConfig, ArticleIdMap, CustomerIdMap
import torch
import json
from typing import Tuple
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import negative_sampling


def shuffle_data(data: Data) -> Data:
    new_edge_order = torch.randperm(data.edge_label.size(0))
    data.edge_label = data.edge_label[new_edge_order]
    data.edge_label_index = data.edge_label_index[:, new_edge_order]
    return data


def create_dataloaders_homo(
    config: DataLoaderConfig,
) -> Tuple[
    LinkNeighborLoader,
    LinkNeighborLoader,
    LinkNeighborLoader,
    CustomerIdMap,
    ArticleIdMap,
]:
    data = torch.load("data/derived/graph.pt")
    # Add a reverse ('article', 'rev_buys', 'customer') relation for message passing:
    data = T.ToUndirected()(data)

    transform = RandomLinkSplit(
        is_undirected=True,
        add_negative_train_samples=True,
        num_val=config.val_split,
        num_test=config.test_split,
        neg_sampling_ratio=0.5,
    )
    train_split, val_split, test_split = transform(data)

    # Confirm that every node appears in every set above
    assert (
        train_split.num_nodes == val_split.num_nodes
        and train_split.num_nodes == test_split.num_nodes
    )

    customer_id_map = read_json("data/derived/customer_id_map_forward.json")
    article_id_map = read_json("data/derived/article_id_map_forward.json")

    train_loader = LinkNeighborLoader(
        shuffle_data(train_split),
        batch_size=config.batch_size,
        num_neighbors=[10, 10],
        # shuffle=True, # This is not yet implemented in the source code
        directed=False,
        edge_label_index=train_split.edge_label_index,
        edge_label=train_split.edge_label,
    )

    val_loader = LinkNeighborLoader(
        shuffle_data(val_split),
        batch_size=config.batch_size,
        num_neighbors=[10, 10],
        # shuffle=True, # This is not yet implemented in the source code
        directed=False,
        edge_label_index=val_split.edge_label_index,
        edge_label=val_split.edge_label,
    )

    test_loader = LinkNeighborLoader(
        shuffle_data(test_split),
        batch_size=config.batch_size,
        num_neighbors=[10, 10],
        # shuffle=True, # This is not yet implemented in the source code
        directed=False,
        edge_label_index=test_split.edge_label_index,
        edge_label=test_split.edge_label,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        customer_id_map,
        article_id_map,
        data,
    )


def create_datasets_homo(
    config: DataLoaderConfig,
) -> Tuple[Data, Data, Data, CustomerIdMap, ArticleIdMap]:
    data = torch.load("data/derived/graph.pt")
    # Add a reverse ('article', 'rev_buys', 'customer') relation for message passing:
    undirected_transformer = T.ToUndirected()
    data = undirected_transformer(data)

    transform = RandomLinkSplit(
        is_undirected=True,
        add_negative_train_samples=False,
        num_val=config.val_split,
        num_test=config.test_split,
        neg_sampling_ratio=0,
    )
    train_split, val_split, test_split = transform(data)

    # Confirm that every node appears in every set above
    assert (
        train_split.num_nodes == val_split.num_nodes
        and train_split.num_nodes == test_split.num_nodes
    )

    customer_id_map = read_json("data/derived/customer_id_map_forward.json")
    article_id_map = read_json("data/derived/article_id_map_forward.json")

    assert torch.max(train_split.edge_stores[0].edge_index) <= train_split.num_nodes

    return (train_split, val_split, test_split, customer_id_map, article_id_map, data)


def read_json(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)
