import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_networkx

from typing import Union

val_map = {"0": 1.0, "1": 0.35}


def visualize_graph(graph: Union[Data, HeteroData]) -> None:
    fig = plt.figure(1, figsize=(25, 25), dpi=60)
    # Convert Graph to NetworkX
    if type(graph) == HeteroData:
        features = graph.collect("x")
        graph = graph.to_homogeneous()

    node_types = graph.node_type
    G = to_networkx(graph)

    # Create Colors for Nodes and Edges
    # values = [val_map.get(node, 0.25) for node in G.nodes()]

    values = [val_map.get(str(node_type.item()), 0.25) for node_type in node_types]

    # Specify layout
    # pos = nx.bipartite_layout(
    #     G,
    #     nodes=[
    #         node if node_type == 0 else None
    #         for node, node_type in zip(G.nodes(), node_types)
    #     ],
    # )
    pos = nx.kamada_kawai_layout(G)
    # pos = nx.spring_layout(G)

    # nx.draw(G, node_size=30, pos=nx.spring_layout(G))
    nx.draw_networkx_nodes(
        G, pos, cmap=plt.get_cmap("jet"), node_color=values, node_size=35
    )
    # nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color="b", arrows=True)
    # nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)
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
