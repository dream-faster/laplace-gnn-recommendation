from torch_geometric.data import HeteroData
from utils.constants import Constants
import torch
from tests.test_utils import get_raw_sample, get_raw_all


def get_raw_data():
    node_features = torch.stack(
        [torch.tensor([0.3, 0.2]), torch.tensor([1.8, 1.9]), torch.tensor([2.8, 2.9])]
    )
    article_features = torch.stack(
        [
            torch.tensor([0.2, 0.3, 0.4, 0.5]),
            torch.tensor([1.2, 1.4, 1.5, 1.6]),
            torch.tensor([2.1, 2.2, 2.3, 2.4]),
            torch.tensor([3.1, 3.2, 3.3, 3.4]),
            torch.tensor([4.1, 4.2, 4.3, 4.4]),
        ]
    )
    graph_edges = torch.tensor([[0, 0, 0, 1, 2], [0, 2, 4, 1, 3]])

    sampled_edges = torch.tensor([[0, 0], [2, 3]])
    labels = torch.tensor([1, 0])

    return node_features, article_features, graph_edges, sampled_edges, labels


def create_dummy_data(save=False):
    node_features, article_features, graph_edges, sampled_edges, labels = get_raw_data()

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

    edges_dict = {0: [0, 2], 1: [1]}
    rev_edges_dict = {0: [0], 1: [1], 2: [0]}

    raw_data = get_raw_sample(data)
    raw_all = get_raw_all(data)

    if save:
        torch.save(data, "data/derived/dummy_graph_train.pt")
        torch.save(edges_dict, "data/derived/dummy_edges_train.pt")
        torch.save(rev_edges_dict, "data/derived/dummy_rev_edges_train.pt")

    return data, raw_data, raw_all


def save_dummy_data():
    create_dummy_data(save=True)
