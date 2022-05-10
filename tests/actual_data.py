from torch_geometric.data import Data, HeteroData
from utils.constants import Constants
import torch
from data.dataset import GraphDataset
from config import link_pred_config
from tests.util import get_raw_sample


def create_data():
    data_dir = "data/derived/"
    train_dataset = GraphDataset(
        config=link_pred_config.dataloader_config,
        edge_path=data_dir + "edges_train.pt",
        graph_path=data_dir + "train_graph.pt",
        train=True,
    )

    first_data = train_dataset[0]

    raw_data = get_raw_sample(first_data.graph)
    return first_data, raw_data
