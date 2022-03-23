#%%

import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data as GeoData
from data.dataset import FashionDataset
from utils.types import DataLoaderConfig
from utils.data import read_graph, graph_to_GeoData


def train_test_val_split(
    data: GeoData, config: DataLoaderConfig
) -> tuple[GeoData, GeoData, GeoData]:
    print("| Splitting the graph into train, val and test")
    transform = RandomLinkSplit(
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=0,
        num_val=config.val_split,
        num_test=config.test_split,
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


config = DataLoaderConfig(test_split=0.15, val_split=0.15)


def run_dataloader(
    config: DataLoaderConfig,
) -> tuple[
    tuple[FashionDataset, GeoData],
    tuple[FashionDataset, GeoData],
    tuple[FashionDataset, GeoData],
]:
    # %%
    graph = read_graph("data/saved/graph.gpickle")
    data = graph_to_GeoData(graph)
    train_split, val_split, test_split = train_test_val_split(data, config)

    train_ev = FashionDataset("temp", edge_index=train_split.edge_label_index)
    train_mp = GeoData(edge_index=train_split.edge_index)

    val_ev = FashionDataset("temp", edge_index=val_split.edge_label_index)
    val_mp = GeoData(edge_index=val_split.edge_index)

    test_ev = FashionDataset("temp", edge_index=test_split.edge_label_index)
    test_mp = GeoData(edge_index=test_split.edge_index)

    return (train_ev, train_mp), (val_ev, val_mp), (test_ev, test_mp)


if __name__ == "__main__":
    run_dataloader(config)
