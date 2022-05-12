from torch_geometric.data import HeteroData
from utils.constants import Constants
import torch as t
from utils.types import NodeFeatures, ArticleFeatures, AllEdges, SampledEdges, Labels
from typing import Optional
from .util import construct_heterodata, deconstruct_heterodata, get_edge_dicts
from torch import Tensor
from tests.types import GeneratorConfig


def create_entire_graph_data(
    save=False, generated=False, config: Optional[GeneratorConfig] = None
) -> HeteroData:
    (
        node_features,
        article_features,
        graph_edges,
    ) = __get_raw_data(generated=generated, config=config)

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

    """ Get Edges """
    subgraph_edges = t.cat(
        [edge_index[:, edge_index[0] == user_id] for user_id in user_ids], dim=1
    )

    # Here we sample as positive edges: 0, the biggest connected edge and the last edge in the entire graph as negative edges
    sampled_edges = t.tensor(
        [
            [0, 0, 0],
            [min(edges_dict[0]), max(edges_dict[0]), t.max(edge_index[1]).item()],
        ],
        dtype=t.long,
    )
    labels = t.tensor([1, 1, 0], dtype=t.long)

    """ Get Features """
    user_features, article_features = (
        user_features[user_ids],
        article_features[t.unique(t.cat([art_ids, sampled_edges[1]]))],
    )

    """ Remap to 0 """
    for i in [0, 1]:
        buckets = t.unique(t.cat([subgraph_edges[i], sampled_edges[i]]))
        subgraph_edges[i] = t.bucketize(subgraph_edges[i], boundaries=buckets)
        sampled_edges[i] = t.bucketize(sampled_edges[i], boundaries=buckets)

    """ Create Subgraph Heterodata """
    subgraph = construct_heterodata(
        user_features, article_features, subgraph_edges, sampled_edges, labels
    )

    return subgraph


def __get_raw_data(
    generated: bool = False,
    config: Optional[GeneratorConfig] = None,
) -> tuple[NodeFeatures, ArticleFeatures, AllEdges]:
    if generated and config is not None:
        return __generated(
            config.num_users,
            config.num_user_features,
            config.num_articles,
            config.num_article_features,
            config.connection_ratio,
        )
    else:
        return __manual()


def __manual() -> tuple[NodeFeatures, ArticleFeatures, AllEdges]:
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
    graph_edges = t.tensor([[0, 0, 0, 1, 1, 2, 2], [0, 2, 4, 1, 5, 3, 0]]).type(t.long)

    return node_features, article_features, graph_edges


def __generated(
    num_users: int = 3,
    num_user_features: int = 2,
    num_articles: int = 10,
    num_article_features: int = 5,
    connection_ratio: float = 0.5,
) -> tuple[NodeFeatures, ArticleFeatures, AllEdges]:
    node_features = t.stack(
        [
            t.tensor([i + float(f"0.{j}") for j in range(num_user_features)])
            for i in range(num_users)
        ]
    )
    article_features = t.stack(
        [
            t.tensor([i + float(f"0.{j}") for j in range(num_article_features)])
            for i in range(num_articles)
        ]
    )

    num_connections = int((num_users * num_articles) * connection_ratio)

    graph_edges = t.stack(
        [
            t.randint(low=0, high=num_users, size=(num_connections,)),
            t.randint(low=0, high=num_articles, size=(num_connections,)),
        ],
    ).type(t.long)

    return node_features, article_features, graph_edges
