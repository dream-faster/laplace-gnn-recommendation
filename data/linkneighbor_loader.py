from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import HeteroData
from data.types import DataLoaderConfig, ArticleIdMap, CustomerIdMap
import torch
import json
from typing import Tuple
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader


def shuffle_data(data: HeteroData) -> HeteroData:
    new_edge_order = torch.randperm(
        data[("customer", "buys", "article")].edge_label.size(0)
    )
    data[("customer", "buys", "article")].edge_label = data[
        ("customer", "buys", "article")
    ].edge_label[new_edge_order]
    data[("customer", "buys", "article")].edge_label_index = data[
        ("customer", "buys", "article")
    ].edge_label_index[:, new_edge_order]

    return data


def create_dataloaders(
    config: DataLoaderConfig,
) -> Tuple[
    LinkNeighborLoader,
    LinkNeighborLoader,
    LinkNeighborLoader,
    CustomerIdMap,
    ArticleIdMap,
]:
    data = torch.load("data/derived/graph_pyg.pt")
    # Add a reverse ('article', 'rev_buys', 'customer') relation for message passing:
    data = T.ToUndirected()(data)

    # Perform a link-level split into training, validation, and test edges:
    train_split, val_split, test_split = T.RandomLinkSplit(
        num_val=config.val_split,
        num_test=config.test_split,
        neg_sampling_ratio=0.5,
        add_negative_train_samples=True,
        edge_types=[("customer", "buys", "article")],
        rev_edge_types=[("article", "rev_buys", "customer")],
        is_undirected=True,
    )(data)
    # when neg_sampling_ratio > 0 and add_negative_train_samples=True only then you will have negative edges

    train_loader = LinkNeighborLoader(
        train_split,
        num_neighbors=[config.num_neighbors] * config.num_neighbors_it,
        batch_size=config.batch_size,
        edge_label_index=(
            ("customer", "buys", "article"),
            train_split[("customer", "buys", "article")].edge_label_index,
        ),
        edge_label=train_split[("customer", "buys", "article")].edge_label,
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
            ("customer", "buys", "article"),
            val_split[("customer", "buys", "article")].edge_label_index,
        ),
        edge_label=val_split[("customer", "buys", "article")].edge_label,
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
            ("customer", "buys", "article"),
            test_split[("customer", "buys", "article")].edge_label_index,
        ),
        edge_label=test_split[("customer", "buys", "article")].edge_label,
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