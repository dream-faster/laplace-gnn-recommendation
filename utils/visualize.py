from distutils.util import convert_path
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_networkx
from torch import Tensor
import torch

from typing import Tuple, Union, Optional, List
from utils.constants import Constants

val_map = {Constants.node_user: 1.0, Constants.node_item: 0.35}


def select_layout(G: Union[nx.DiGraph, nx.Graph], node_types: Optional[list]):
    one_category_nodes = [
        node if node_type == Constants.node_user else None
        for node, node_type in zip(G.nodes(), node_types)
    ]

    pos = nx.bipartite_layout(G, nodes=one_category_nodes)
    # pos = nx.kamada_kawai_layout(G)
    # pos = nx.planar_layout(G)
    # pos = nx.shell_layout(G)
    # pos = nx.spectral_layout(G)
    # pos = nx.circular_layout(G)
    # pos = nx.spiral_layout(G)
    # pos = nx.spring_layout(G)
    return pos


def get_edges(
    data: HeteroData,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    _, edge_types = data.metadata()

    edge_dicts = [data[edge_type] for edge_type in edge_types]

    all_edges = edge_dicts[0].edge_index.detach().clone()
    edge_label = edge_dicts[0].edge_label.detach().clone()
    edge_label_index = edge_dicts[0].edge_label_index.detach().clone()

    len_customer_indexes = torch.max(all_edges, dim=1)[0][0].item() + 1
    edge_label_index[1] = torch.add(edge_label_index[1], len_customer_indexes)
    all_edges[1] = torch.add(all_edges[1], len_customer_indexes)

    negative_edges_sampled: Tensor = edge_label_index[:, edge_label == 0]
    positive_edges_sampled: Tensor = edge_label_index[:, edge_label == 1]

    return (
        negative_edges_sampled.t().contiguous().tolist(),
        positive_edges_sampled.t().contiguous().tolist(),
        all_edges.t().contiguous().tolist(),
    )


def manual_graph(data: Union[Data, HeteroData]) -> Union[nx.Graph, nx.DiGraph]:
    G = nx.DiGraph()
    negative_edges, positive_edges, all_edges = get_edges(data)

    customer_nodes = data[Constants.node_user].x
    article_nodes = data[Constants.node_item].x

    [
        G.add_node(i, x=x.tolist(), node_type=Constants.node_user)
        for i, x in enumerate(customer_nodes)
    ]
    [
        G.add_node(i + len(customer_nodes), x=x.tolist(), node_type=Constants.node_item)
        for i, x in enumerate(article_nodes)
    ]

    G.add_edges_from([edge_pair for edge_pair in all_edges])

    return G, negative_edges, positive_edges


def visualize_graph(data: Union[Data, HeteroData]) -> None:
    fig = plt.figure(1, figsize=(25, 14), dpi=60)

    """ Convert Graph to NetworkX Manually """
    G, negative_edges, positive_edges = manual_graph(data)

    """ Create list of node types to be able to color nodes """
    node_types = [node[1]["node_type"] for node in G.nodes(data=True)]
    values = [val_map.get(node_type, 0.25) for node_type in node_types]

    """ Specify node layout """
    pos = select_layout(G, node_types)

    """ Add nodes """
    nx.draw_networkx_nodes(
        G,
        pos,
        cmap=plt.get_cmap("PiYG"),
        node_color=values,
        node_size=165,
    )

    """ Add labels """
    nx.draw_networkx_labels(
        G,
        pos,
        labels={i: str(i) + "\n" + k for i, k in enumerate(node_types)},
        font_size=12,
        font_color="black",
    )

    """ Add all edges """
    # First we have to draw all edges otherwise it overrides the edges after
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=G.edges(),
        edge_color="grey",
        # connectionstyle="arc3, rad = 0.1",
        arrows=True,
        style="solid",
        alpha=0.20,
    )

    """ Add sampled edges """
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=negative_edges,
        edge_color="r",
        connectionstyle="arc3, rad = 0.01",
        arrows=True,
        style="dashed",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=positive_edges,
        edge_color="g",
        connectionstyle="arc3, rad = -0.01",
        arrows=True,
        style="solid",
    )
    plt.show()
