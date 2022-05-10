from torch_geometric.data import Data, HeteroData
from utils.constants import Constants
import torch
from data.dataset import GraphDataset
from config import link_pred_config
from tests.test_utils import get_raw_sample


def create_data():
    data_dir = "data/derived/"
    train_dataset = GraphDataset(
        config=link_pred_config,
        edge_path=data_dir + "dummy_edges_train.pt",
        graph_path=data_dir + "dummy_graph_train.pt",
        article_edge_path=data_dir + "dummy_rev_edges_train.pt",
        train=True,
    )

    first_data = train_dataset[0]

    return first_data
