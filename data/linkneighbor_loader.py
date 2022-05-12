from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import HeteroData
from data.types import ArticleIdMap, CustomerIdMap
from config import Config
import torch as t
import json
from typing import Tuple
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from utils.constants import Constants


def shuffle_data(data: HeteroData) -> HeteroData:
    new_edge_order = t.randperm(data[Constants.edge_key].edge_label.size(0))
    data[Constants.edge_key].edge_label = data[Constants.edge_key].edge_label[
        new_edge_order
    ]
    data[Constants.edge_key].edge_label_index = data[
        Constants.edge_key
    ].edge_label_index[:, new_edge_order]

    return data


def create_dataloaders(
    config: Config,
) -> Tuple[
    LinkNeighborLoader,
    LinkNeighborLoader,
    LinkNeighborLoader,
    CustomerIdMap,
    ArticleIdMap,
]:
    data = t.load("data/derived/graph_pyg.pt")
    # Add a reverse ('article', 'rev_buys', 'customer') relation for message passing:
    data = T.ToUndirected()(data)

    # Perform a link-level split into training, validation, and test edges:
    train_split, val_split, test_split = T.RandomLinkSplit(
        num_val=config.val_split,
        num_test=config.test_split,
        neg_sampling_ratio=0.5,
        add_negative_train_samples=True,
        edge_types=[Constants.edge_key],
        rev_edge_types=[(Constants.node_item, "rev_buys", Constants.node_user)],
        is_undirected=True,
    )(data)
    # when neg_sampling_ratio > 0 and add_negative_train_samples=True only then you will have negative edges

    train_loader = LinkNeighborLoader(
        train_split,
        num_neighbors=[config.num_neighbors] * config.num_neighbors_it,
        batch_size=config.batch_size,
        edge_label_index=(
            Constants.edge_key,
            train_split[Constants.edge_key].edge_label_index,
        ),
        edge_label=train_split[Constants.edge_key].edge_label,
        directed=False,
        replace=False,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = LinkNeighborLoader(
        val_split,
        num_neighbors=[64] * 2,
        batch_size=config.batch_size,
        edge_label_index=(
            Constants.edge_key,
            val_split[Constants.edge_key].edge_label_index,
        ),
        edge_label=val_split[Constants.edge_key].edge_label,
        directed=False,
        replace=False,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = LinkNeighborLoader(
        test_split,
        num_neighbors=[64] * 2,
        batch_size=config.batch_size,
        edge_label_index=(
            Constants.edge_key,
            test_split[Constants.edge_key].edge_label_index,
        ),
        edge_label=test_split[Constants.edge_key].edge_label,
        directed=False,
        replace=False,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    customer_id_map = read_json("data/derived/customer_id_map_forward.json")
    article_id_map = read_json("data/derived/article_id_map_forward.json")

    return (
        train_loader,
        val_loader,
        test_loader,
        customer_id_map,
        article_id_map,
        data,
    )


def read_json(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)
