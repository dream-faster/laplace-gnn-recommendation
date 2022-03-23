#%%
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import RandomLinkSplit
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


def train_test_val_split(data: GeoData) -> tuple[GeoData, GeoData, GeoData]:
    print("| Splitting the graph into train and test")
    # Train/val/test split
    transform = RandomLinkSplit(
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=0,
        num_val=0.15,
        num_test=0.15,
    )
    train_split, val_split, test_split = transform(data)

    # Confirm that every node appears in every set above
    assert (
        train_split.num_nodes == val_split.num_nodes
        and train_split.num_nodes == test_split.num_nodes
    )
    print(train_split)
    print(val_split)
    print(test_split)
    return train_split, val_split, test_split


# %%
graph = read_graph("data/graph.gpickle")
data = graph_to_GeoData(graph)
train_split, val_split, test_split = train_test_val_split(data)
