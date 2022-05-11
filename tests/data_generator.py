from numpy import column_stack
from torch_geometric.data import HeteroData
from utils.constants import Constants
import torch as t
from torch import Tensor
from .util import get_raw_sample, get_raw_all
from utils.types import NodeFeatures, ArticleFeatures, AllEdges, SampledEdges, Labels
import pandas as pd


def extract_edges(transactions: pd.DataFrame, by: str, get: str) -> dict:
    return transactions.groupby(by)[get].apply(list).to_dict()


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

    graph_edges_pd = (
        pd.DataFrame(graph_edges.numpy())
        .transpose()
        .rename({0: "user", 1: "article"}, axis=1)
    )
    edges_dict = extract_edges(
        graph_edges_pd,
        by="user",
        get="article",
    )
    rev_edges_dict = extract_edges(
        graph_edges_pd,
        by="article",
        get="user",
    )

    subgraph_data = __create_final_subgraph(data, edges_dict, rev_edges_dict, n_hop=2)
    raw_data = get_raw_sample(data)
    raw_all = get_raw_all(data)

    if save:
        t.save(subgraph_data, "data/derived/dummy_graph_train.pt")
        t.save(edges_dict, "data/derived/dummy_edges_train.pt")
        t.save(rev_edges_dict, "data/derived/dummy_rev_edges_train.pt")

    return subgraph_data, raw_data, raw_all


def save_dummy_data():
    create_dummy_data(save=True)


def __deconstruct_heterodata(
    hdata: HeteroData,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        hdata[Constants.node_user].x,
        hdata[Constants.node_item].x,
        hdata[Constants.edge_key].edge_index,
    )


def __create_final_subgraph(
    entire_graph: HeteroData, edges_dict: dict, rev_edges_dict: dict, n_hop: int
) -> HeteroData:
    user_ids = t.tensor([0], dtype=t.long)
    art_ids = t.empty(0, dtype=t.long)

    """ Collect ids """
    for _ in range(n_hop):
        art_ids = t.unique(
            t.cat(
                [
                    art_ids,
                    t.tensor([i for id in user_ids for i in edges_dict[id.item()]]),
                ]
            )
        )
        user_ids = t.unique(
            t.cat(
                [
                    user_ids,
                    t.tensor([i for id in art_ids for i in rev_edges_dict[id.item()]]),
                ]
            )
        )

    user_features, article_features, edge_index = __deconstruct_heterodata(entire_graph)

    """ Get Features """
    user_features, article_features = (
        user_features[user_ids],
        article_features[art_ids],
    )

    """ Get Edges """
    subgraph_edges = t.cat(
        [edge_index[:, edge_index[0] == user_id] for user_id in user_ids], dim=1
    )

    # Here we sample as positive edges: 0, the biggest connected edge and the last edge in the entire graph as negative edges
    sampled_edges = t.tensor(
        [[0, 0, 0], [0, max(edges_dict[0]), t.max(edge_index[1]).item()]],
        dtype=t.long,
    )
    labels = t.tensor([1, 1, 0])

    """Create Data"""
    subgraph = HeteroData()
    subgraph[Constants.node_user].x = user_features
    subgraph[Constants.node_item].x = article_features

    # Add original directional edges
    subgraph[Constants.edge_key].edge_index = subgraph_edges
    subgraph[Constants.edge_key].edge_label_index = sampled_edges
    subgraph[Constants.edge_key].edge_label = labels

    # Add reverse edges
    reverse_key = t.LongTensor([1, 0])
    subgraph[Constants.rev_edge_key].edge_index = subgraph_edges[reverse_key]
    subgraph[Constants.rev_edge_key].edge_label_index = sampled_edges[reverse_key]
    subgraph[Constants.rev_edge_key].edge_label = labels

    return subgraph


def __get_raw_data() -> tuple[
    NodeFeatures, ArticleFeatures, AllEdges, SampledEdges, Labels
]:
    node_features = t.stack(
        [t.tensor([0.0, 0.1]), t.tensor([1.0, 1.1]), t.tensor([2.0, 2.1])]
    )
    article_features = t.stack(
        [
            t.tensor([0.0, 0.1, 0.2, 0.3, 0.4]),
            t.tensor([1.0, 1.1, 1.2, 1.3, 1.4]),
            t.tensor([2.0, 2.1, 2.2, 2.3, 2.4]),
            t.tensor([3.0, 3.1, 3.2, 3.3, 3.4]),
            t.tensor([4.0, 4.1, 4.2, 4.3, 4.4]),
            t.tensor([5.0, 5.1, 5.2, 5.3, 5.4]),
        ]
    )
    graph_edges = t.tensor([[0, 0, 0, 1, 1, 2, 2], [0, 2, 4, 1, 5, 3, 0]])

    # Last sample: We simulate how the last edge is taken from the larger graph, 5 > len(article_features)
    sampled_edges = t.tensor([[0, 0, 0], [2, 3, 5]])
    labels = t.tensor([1, 0, 0])

    return node_features, article_features, graph_edges, sampled_edges, labels
