from torch_geometric.data import HeteroData
from utils.constants import Constants
import torch as t
from utils.types import NodeFeatures, ArticleFeatures, AllEdges, SampledEdges, Labels
from typing import Optional
from .util import deconstruct_heterodata, get_edge_dicts
from torch import Tensor


def create_dummy_data(save=False) -> HeteroData:
    (
        node_features,
        article_features,
        graph_edges,
    ) = __get_raw_data()

    """Create Data"""
    data = HeteroData()
    data[Constants.node_user].x = node_features
    data[Constants.node_item].x = article_features
    data[Constants.edge_key].edge_index = graph_edges

    edges_dict, rev_edges_dict = get_edge_dicts(data[Constants.edge_key].edge_index)

    if save:
        t.save(data, "data/derived/dummy_graph_train.pt")
        t.save(edges_dict, "data/derived/dummy_edges_train.pt")
        t.save(rev_edges_dict, "data/derived/dummy_rev_edges_train.pt")

    return data


def create_subgraph_comparison(n_hop: int) -> HeteroData:
    """Load Data"""
    entire_graph = t.load("data/derived/dummy_graph_train.pt")
    edges_dict = t.load("data/derived/dummy_edges_train.pt")
    rev_edges_dict = t.load("data/derived/dummy_rev_edges_train.pt")

    """Preprocess entire graph"""
    user_features, article_features, edge_index, _, _ = deconstruct_heterodata(
        entire_graph
    )

    edges_dict, rev_edges_dict = get_edge_dicts(edge_index)

    """ Collect ids """
    user_ids = t.tensor([0], dtype=t.long)
    art_ids = t.empty(0, dtype=t.long)

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

    """ Remap to 0 """
    for i in [0, 1]:
        subgraph_edges[i] = t.bucketize(subgraph_edges[i], t.unique(subgraph_edges[i]))
        sampled_edges[i] = t.bucketize(
            sampled_edges[i], t.unique(t.cat([subgraph_edges[i], sampled_edges[i]]))
        )

    """ Create Subgraph Heterodata """
    subgraph = __construct_heterodata(
        user_features, article_features, subgraph_edges, sampled_edges, labels
    )

    return subgraph


def __construct_heterodata(
    user_features: NodeFeatures,
    article_features: ArticleFeatures,
    edge_index: AllEdges,
    edge_label_index: Optional[SampledEdges],
    edge_label: Optional[Labels],
) -> HeteroData:

    """Create Data"""
    data = HeteroData()
    data[Constants.node_user].x = user_features
    data[Constants.node_item].x = article_features

    # Add original directional edges
    data[Constants.edge_key].edge_index = edge_index
    if type(edge_label_index) is SampledEdges:
        data[Constants.edge_key].edge_label_index = edge_label_index
    if type(edge_label) is Labels:
        data[Constants.edge_key].edge_label = edge_label

    # Add reverse edges
    reverse_key = t.LongTensor([1, 0])
    data[Constants.rev_edge_key].edge_index = edge_index[reverse_key]
    if type(edge_label_index) is SampledEdges:
        data[Constants.rev_edge_key].edge_label_index = edge_label_index[reverse_key]
    if type(edge_label) is Labels:
        data[Constants.rev_edge_key].edge_label = edge_label

    return data


def __get_raw_data() -> tuple[NodeFeatures, ArticleFeatures, AllEdges]:
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

    return node_features, article_features, graph_edges
