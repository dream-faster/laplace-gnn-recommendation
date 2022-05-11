from torch_geometric.data import HeteroData
from config import Config
from data.types import ArticleIdMap, CustomerIdMap
import torch
import json
from typing import Tuple
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader, DataLoader
from .dataset import GraphDataset
from .matching.lightgcn import LightGCNMatcher
from .matching.users_with_common_purchases import UsersWithCommonPurchasesMatcher
from .matching.users_same_location import UsersSameLocationMatcher
from .matching.popular_items import PopularItemsMatcher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataloaders(
    config: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader, CustomerIdMap, ArticleIdMap, HeteroData]:
    data_dir = "data/derived/"
    train_dataset = GraphDataset(
        config=config,
        graph_path=data_dir + "train_graph.pt",
        users_adj_list=data_dir + "edges_train.pt",
        articles_adj_list=data_dir + "rev_edges_train.pt",
        train=True,
    )
    val_dataset = GraphDataset(
        config=config,
        graph_path=data_dir + "val_graph.pt",
        users_adj_list=data_dir + "edges_val.pt",
        articles_adj_list=data_dir + "rev_edges_val.pt",
        train=False,
        matchers=[
            # LightGCNMatcher(config.candidate_pool_size),
            PopularItemsMatcher(config.candidate_pool_size),
            # UsersSameLocationMatcher(config.candidate_pool_size, "val"),
            UsersWithCommonPurchasesMatcher(config.candidate_pool_size, "val"),
        ],
    )
    test_dataset = GraphDataset(
        config=config,
        graph_path=data_dir + "test_graph.pt",
        users_adj_list=data_dir + "edges_test.pt",
        articles_adj_list=data_dir + "rev_edges_test.pt",
        train=False,
        matchers=[
            # LightGCNMatcher(config.candidate_pool_size),
            PopularItemsMatcher(config.candidate_pool_size),
            # UsersSameLocationMatcher(config.candidate_pool_size, "test"),
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
