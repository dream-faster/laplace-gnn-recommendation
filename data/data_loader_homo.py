from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
from data.types import DataLoaderConfig, ArticleIdMap, CustomerIdMap
import torch
import json
from typing import Tuple
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader


def create_dataloaders_homo(
    config: DataLoaderConfig,
) -> Tuple[
    NeighborLoader,
    NeighborLoader,
    NeighborLoader,
    CustomerIdMap,
    ArticleIdMap,
]:
    data = torch.load("data/derived/graph.pt")
    # Add a reverse ('article', 'rev_buys', 'customer') relation for message passing:
    data = T.ToUndirected()(data)

    # from torch_geometric.datasets import MovieLens
    # import os.path as osp

    # path = osp.join(osp.dirname(osp.realpath(__file__)), "data/MovieLens")
    # data_mov = MovieLens(path, model_name="all-MiniLM-L6-v2")[0]

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

    return (
        NeighborLoader(
            train_split,
            batch_size=config.batch_size,
            num_neighbors=[10, 10],
            shuffle=True,
        ),
        NeighborLoader(
            val_split,
            batch_size=config.batch_size,
            num_neighbors=[10, 10],
            shuffle=True,
        ),
        NeighborLoader(
            test_split,
            batch_size=config.batch_size,
            num_neighbors=[10, 10],
            shuffle=True,
        ),
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
