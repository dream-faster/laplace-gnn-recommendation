from torch_geometric.data import HeteroData
from utils.constants import Constants
import torch as t
from tests.data_generator import create_entire_graph_data, create_subgraph_comparison
from torch_geometric import seed_everything
from tests.util import (
    get_first_item_from_dataset,
    deconstruct_heterodata,
    preprocess_and_load_to_neo4j,
)
from tests.types import GeneratorConfig, generator_config
from config import link_pred_config
import pandas as pd

seed_everything(5)
# Generate and save entire graph data:
original_data = create_entire_graph_data(
    save=True, config=generator_config, type="random"
)

# This is the data we are comparing it to:
data_comparison = create_subgraph_comparison(n_hop=link_pred_config.n_hop_neighbors)


def test_integrity_base(graph_database: bool = False):
    if graph_database:
        preprocess_and_load_to_neo4j(original_data)

    # This is the data we are testing:
    data_from_dataset = get_first_item_from_dataset(graph_database=graph_database)

    integrity_edges(data=data_from_dataset, data_comp=data_comparison)
    integrity_nodes(data=data_from_dataset, data_comp=data_comparison)


def integrity_edges(data: HeteroData, data_comp: HeteroData = data_comparison):
    for edge_type in [Constants.edge_key, Constants.rev_edge_key]:
        edges = data[edge_type]

        # Basic Testing of edges if they fit size expectations
        assert edges.edge_index.shape[0] == 2
        assert edges.edge_index.shape[1] > 0
        assert edges.edge_label_index.shape[0] == 2
        assert edges.edge_label_index.shape[1] > 0
        assert edges.edge_label.shape[0] > 0
        assert len(edges.edge_label.shape) == 1
        assert edges.edge_label.shape[0] == edges.edge_label_index.shape[1]

        # Comparing to subgraph dataset that we construct
        edges_comp = data_comp[edge_type]
        assert t.equal(
            edges.edge_index[0].sort()[0], edges_comp.edge_index[0].sort()[0]
        )
        assert t.equal(
            edges.edge_index[1].sort()[0], edges_comp.edge_index[1].sort()[0]
        )
        assert t.equal(
            edges.edge_label_index[0].sort()[0],
            edges_comp.edge_label_index[0].sort()[0],
        )
        assert t.equal(
            edges.edge_label_index[1].sort()[0],
            edges_comp.edge_label_index[1].sort()[0],
        )
        assert t.equal(edges.edge_label, edges_comp.edge_label)


def integrity_nodes(data: HeteroData, data_comp: HeteroData = data_comparison):
    (
        user_features,
        article_features,
        edge_index,
        edge_label_index,
        edge_label,
    ) = deconstruct_heterodata(data)

    for node_type in [Constants.node_user, Constants.node_item]:
        # Comparing to subgraph dataset that we construct
        nodes = data[node_type]
        nodes_comp = data_comp[node_type]
        assert t.equal(nodes.x, nodes_comp.x)

    all_touched_users = t.unique(t.concat([edge_index[0], edge_label_index[0]]))
    all_touched_articles = t.unique(t.concat([edge_index[1], edge_label_index[1]]))

    assert (
        user_features.shape[0] == all_touched_users.shape[0]
    ), "User features are not the same as existing and sampled edges."

    assert (
        article_features.shape[0] == all_touched_articles.shape[0]
    ), "Article features are not the same as existing and sampled edges."
