from torch_geometric.data import HeteroData
from utils.constants import Constants
import torch
from torch import Tensor
from .util import get_raw_sample, get_raw_all
from utils.types import NodeFeatures, ArticleFeatures, AllEdges, SampledEdges, Labels


def create_dummy_data(save=False):
    (
        node_features,
        article_features,
        graph_edges,
        sampled_edges,
        labels,
    ) = __get_raw_data()

    """Create Data"""
    data = HeteroData()
    data[Constants.node_user].x = node_features
    data[Constants.node_item].x = article_features

    data[Constants.edge_key].edge_index = graph_edges

    edges_dict = {0: [0, 2], 1: [1]}
    rev_edges_dict = {0: [0], 1: [1], 2: [0]}

    subgraph_data = create_final_subgraph(data, edges_dict, rev_edges_dict, n_hop=2)
    raw_data = get_raw_sample(data)
    raw_all = get_raw_all(data)

    if save:
        torch.save(subgraph_data, "data/derived/dummy_graph_train.pt")
        torch.save(edges_dict, "data/derived/dummy_edges_train.pt")
        torch.save(rev_edges_dict, "data/derived/dummy_rev_edges_train.pt")

    return subgraph_data, raw_data, raw_all


def save_dummy_data():
    create_dummy_data(save=True)


def create_final_subgraph(
    entire_graph: HeteroData, edges_dict: dict, rev_edges_dict: dict, n_hop: int
) -> HeteroData:
    user_nodes = [0]
    user_nodes_list, article_nodes_list = [], []

    for _ in range(n_hop):
        article_nodes = [edges_dict[i] for i in user_nodes]
        user_nodes = [rev_edges_dict[i] for i in article_nodes]
        user_nodes_list.append(user_nodes)
        article_nodes_list.append(article_nodes)

    article_ids = torch.unique(torch.tensor(article_nodes_list))
    user_ids = torch.unique(torch.tensor(user_nodes_list))

    """Create Data"""
    subgraph = HeteroData()
    subgraph[Constants.node_user].x = node_features
    subgraph[Constants.node_item].x = article_features

    # Add original directional edges
    subgraph[Constants.edge_key].edge_index = graph_edges
    subgraph[Constants.edge_key].edge_label_index = sampled_edges
    subgraph[Constants.edge_key].edge_label = labels

    # Add reverse edges
    reverse_key = torch.LongTensor([1, 0])
    subgraph[Constants.rev_edge_key].edge_index = graph_edges[reverse_key]
    subgraph[Constants.rev_edge_key].edge_label_index = sampled_edges[reverse_key]
    subgraph[Constants.rev_edge_key].edge_label = labels

    return subgraph


def __get_raw_data() -> tuple[
    NodeFeatures, ArticleFeatures, AllEdges, SampledEdges, Labels
]:
    node_features = torch.stack(
        [torch.tensor([0.0, 0.1]), torch.tensor([1.0, 1.1]), torch.tensor([2.0, 2.1])]
    )
    article_features = torch.stack(
        [
            torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4]),
            torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4]),
            torch.tensor([2.0, 2.1, 2.2, 2.3, 2.4]),
            torch.tensor([3.0, 3.1, 3.2, 3.3, 3.4]),
            torch.tensor([4.0, 4.1, 4.2, 4.3, 4.4]),
        ]
    )
    graph_edges = torch.tensor([[0, 0, 0, 1, 2], [0, 2, 4, 1, 3]])

    # Last sample: We simulate how the last edge is taken from the larger graph, 5 > len(article_features)
    sampled_edges = torch.tensor([[0, 0, 0], [2, 3, 5]])
    labels = torch.tensor([1, 0, 0])

    return node_features, article_features, graph_edges, sampled_edges, labels
