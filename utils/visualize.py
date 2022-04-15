import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_networkx
from torch import Tensor
import torch

from typing import Union, Optional

val_map = {"0": 1.0, "1": 0.35}


def select_layout(G: Union[nx.DiGraph, nx.Graph], node_types: Optional[list]):
    # pos = nx.bipartite_layout(
    #     G,
    #     nodes=[
    #         node if node_type == 0 else None
    #         for node, node_type in zip(G.nodes(), node_types.tolist())
    #     ],
    # )
    # pos = nx.kamada_kawai_layout(G)
    # pos = nx.planar_layout(G)
    # pos = nx.shell_layout(G)
    # pos = nx.spectral_layout(G)

    # pos = nx.circular_layout(G)
    pos = nx.spiral_layout(G)
    # pos = nx.spring_layout(G)
    return pos


def get_edges(
    G: HeteroData,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    _, edge_types = G.metadata()

    edge_dicts = [G[edge_type] for edge_type in edge_types]

    # all_edges = edge_dicts[0].edge_index
    all_edges = torch.concat([edge_dict.edge_index for edge_dict in edge_dicts], dim=1)

    edge_label_index = edge_dicts[0].edge_label_index
    # edge_label_index = torch.concat(
    #     [edge_dict.edge_label_index for edge_dict in edge_dicts], dim=1
    # )

    edge_label = edge_dicts[0].edge_label
    # edge_label = torch.concat([edge_dict.edge_label for edge_dict in edge_dicts], dim=1)

    negative_edges_sampled: Tensor = edge_label_index[:, edge_label == 0]
    positive_edges_sampled: Tensor = edge_label_index[:, edge_label == 1]

    return (
        negative_edges_sampled.t().tolist(),
        positive_edges_sampled.t().tolist(),
        all_edges.t().tolist(),
    )


def visualize_graph(data: Union[Data, HeteroData]) -> None:
    fig = plt.figure(1, figsize=(25, 14), dpi=60)

    negative_edges, positive_edges, all_edges = get_edges(data)
    # Convert Graph to NetworkX
    if type(data) == HeteroData:
        features = data.collect("x")
        graph = data.to_homogeneous()

    node_types = graph.node_type

    G = to_networkx(graph)

    values = [val_map.get(str(node_type.item()), 0.25) for node_type in node_types]

    # Specify layout
    pos = select_layout(G, node_types)

    nx.draw_networkx_nodes(
        G, pos, cmap=plt.get_cmap("PiYG"), node_color=values, node_size=165
    )
    nx.draw_networkx_labels(G, pos, font_size=7, font_color="white")

    # First we have to draw all edges otherwise it overrides the edges after
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=all_edges,
        edge_color="grey",
        arrows=True,
        style="solid",
        alpha=0.20,
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=negative_edges, edge_color="r", arrows=True, style="dashed"
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=positive_edges, edge_color="g", arrows=True, style="solid"
    )
    plt.show()


def visualize_graph_example() -> None:

    G = nx.DiGraph()
    G.add_edges_from(
        [
            ("A", "B"),
            ("A", "C"),
            ("D", "B"),
            ("E", "C"),
            ("E", "F"),
            ("B", "H"),
            ("B", "G"),
            ("B", "F"),
            ("C", "G"),
        ]
    )

    val_map = {"A": 1.0, "D": 0.5714285714285714, "H": 0.0}

    values = [val_map.get(node, 0.25) for node in G.nodes()]

    # Specify the edges you want here
    red_edges = [("A", "C"), ("E", "C")]
    edge_colours = ["black" if not edge in red_edges else "red" for edge in G.edges()]
    black_edges = [edge for edge in G.edges() if edge not in red_edges]

    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(
        G, pos, cmap=plt.get_cmap("jet"), node_color=values, node_size=500
    )
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color="r", arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)
    plt.show()


if __name__ == "__main__":
    visualize_graph_example()
