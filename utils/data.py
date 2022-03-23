import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data as GeoData


def read_graph(path: str) -> nx.Graph:
    print("| Reading graph from {}".format(path))
    graph = nx.read_gpickle(path)
    nx.draw_spring(graph, node_size=5)
    return graph


def graph_to_GeoData(graph: nx.Graph) -> GeoData:
    print("| Converting NetworkX model to Pytorch Geometric Model")
    data = from_networkx(graph)
    return data
