from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import HeteroData
from data.types import DataLoaderConfig, ArticleIdMap, CustomerIdMap
import torch
import json
from typing import Tuple
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader, DataLoader
from data.dataset import GraphDataset
from data.matching.lightgcn import LightGCNMatcher
from data.matching.users_with_common_purchases import UsersWithCommonPurchasesMatcher
from data.matching.users_same_location import UsersSameLocationMatcher
from data.matching.popular_items import PopularItemsMatcher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataloaders(
    config: DataLoaderConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, CustomerIdMap, ArticleIdMap, HeteroData]:
    data_dir = "data/derived/"
    train_dataset = GraphDataset(
        config=config,
        edge_path=data_dir + "edges_train.pt",
        graph_path=data_dir + "train_graph.pt",
        article_edge_path=data_dir + "rev_edges_train.pt",
        train=True,
    )
    val_dataset = GraphDataset(
        config=config,
        edge_path=data_dir + "edges_val.pt",
        graph_path=data_dir + "val_graph.pt",
        article_edge_path=data_dir + "rev_edges_val.pt",
        train=False,
        matchers=[
            # LightGCNMatcher(config.candidate_pool_size),
            PopularItemsMatcher(config.candidate_pool_size),
            UsersSameLocationMatcher(config.candidate_pool_size, "val"),
            UsersWithCommonPurchasesMatcher(config.candidate_pool_size, "val"),
        ],
    )
    test_dataset = GraphDataset(
        config=config,
        edge_path=data_dir + "edges_test.pt",
        graph_path=data_dir + "test_graph.pt",
        article_edge_path=data_dir + "rev_edges_test.pt",
        train=False,
        matchers=[
            # LightGCNMatcher(config.candidate_pool_size),
            PopularItemsMatcher(config.candidate_pool_size),
            UsersSameLocationMatcher(config.candidate_pool_size, "test"),
            UsersWithCommonPurchasesMatcher(config.candidate_pool_size, "test"),
        ],
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    data = train_dataset.graph
    data = T.ToUndirected()(data)

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
