from torch_geometric.data import HeteroData
from config import Config
from data.types import ArticleIdMap, CustomerIdMap
import torch as t
import json
from typing import Tuple
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader, DataLoader
from .dataset import GraphDataset
from .dataset_neo import GraphDataset as GraphDatasetNeo
from .matching import get_matchers


def create_dataloaders(
    config: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader, CustomerIdMap, ArticleIdMap, HeteroData]:
    Dataset = GraphDatasetNeo if config.neo4j else GraphDataset

    data_dir = "data/derived/"
    train_dataset = Dataset(
        config=config,
        graph_path=data_dir + "train_graph.pt",
        users_adj_list=data_dir + "edges_train.pt",
        articles_adj_list=data_dir + "rev_edges_train.pt",
        train=True,
        split_type="train",
    )

    val_dataset = Dataset(
        config=config,
        graph_path=data_dir + "val_graph.pt",
        users_adj_list=data_dir + "edges_val.pt",
        articles_adj_list=data_dir + "rev_edges_val.pt",
        train=False,
        split_type="val",
        matchers=get_matchers(config.matchers, "val", config.candidate_pool_size),
    )
    test_dataset = Dataset(
        config=config,
        graph_path=data_dir + "test_graph.pt",
        users_adj_list=data_dir + "edges_test.pt",
        articles_adj_list=data_dir + "rev_edges_test.pt",
        train=False,
        split_type="test",
        matchers=get_matchers(config.matchers, "test", config.candidate_pool_size),
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
