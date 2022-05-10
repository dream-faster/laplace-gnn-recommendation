from torch_geometric.data import HeteroData
from utils.constants import Constants
import torch
from tests.test_utils import get_raw_sample


def get_data():
    node_features = torch.stack([torch.tensor([0.3, 0.2]), torch.tensor([0.8, 0.9])])
    article_features = torch.stack(
        [
            torch.tensor([0.2, 0.3, 0.4, 0.5]),
            torch.tensor([1.2, 1.4, 1.5, 1.6]),
            torch.tensor([2.1, 2.2, 2.3, 2.4]),
        ]
    )
    graph_edges = torch.tensor([[0, 0, 1], [0, 2, 1]])
    sampled_edges = torch.tensor([[0, 0], [2, 3]])
    labels = torch.tensor([1, 0])

    return node_features, article_features, graph_edges, sampled_edges, labels


def create_dummy_data():
    node_features, article_features, graph_edges, sampled_edges, labels = get_data()

    """Create Data"""
    data = HeteroData()
    data[Constants.node_user].x = node_features
    data[Constants.node_item].x = article_features

    # Add original directional edges
    data[Constants.edge_key].edge_index = graph_edges
    data[Constants.edge_key].edge_label_index = sampled_edges
    data[Constants.edge_key].edge_label = labels

    # Add reverse edges
    reverse_key = torch.LongTensor([1, 0])
    data[Constants.rev_edge_key].edge_index = graph_edges[reverse_key]
    data[Constants.rev_edge_key].edge_label_index = sampled_edges[reverse_key]
    data[Constants.rev_edge_key].edge_label = labels

    edges_dict = {"0": [0, 2], "1": [1]}
    rev_edges_dict = {"0": [0], "1": [1], "2": [0]}

    raw_data = get_raw_sample(data)

    return data, raw_data
