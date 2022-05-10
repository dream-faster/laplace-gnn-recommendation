from torch_geometric.data import Data, HeteroData
from utils.constants import Constants
import torch
from data.dataset import GraphDataset
from config import link_pred_config


node_1_features = [0.3, 0.2]
node_2_features = [0.8, 0.9]
article_1_features = [0.2, 0.3, 0.4, 0.5]
article_2_features = [1.2, 1.4, 1.5, 1.6]
article_3_features = [2.1, 2.2, 2.3, 2.4]


def get_data():
    node_features = torch.stack([node_1_features, node_2_features])
    article_features = torch.stack(
        [article_1_features, article_2_features, article_3_features]
    )
    graph_edges = torch.tensor([[0, 0, 1], [0, 2, 1]])
    sampled_edges = torch.tensor([[0, 0], [2, 3]])
    labels = torch.tensor([1, 0])

    return node_features, article_features, graph_edges, sampled_edges, labels


def create_dummy_data():
    node_features, article_features, graph_edges, sampled_edges, labels = get_data()

    """Create Data"""
    data = HeteroData()
    data[Constants.node_user].x = node_features
    data[Constants.node_item].x = article_features

    # Add original directional edges
    data[Constants.edge_key].edge_index = graph_edges
    data[Constants.edge_key].edge_label_index = sampled_edges
    data[Constants.edge_key].edge_label = labels

    # Add reverse edges
    reverse_key = torch.LongTensor([1, 0])
    data[Constants.rev_edge_key].edge_index = graph_edges[reverse_key]
    data[Constants.rev_edge_key].edge_label_index = sampled_edges[reverse_key]
    data[Constants.rev_edge_key].edge_label = labels

    edges_dict = {"0": [0, 2], "1": [1]}
    rev_edges_dict = {"0": [0], "1": [1], "2": [0]}

    return data, edges_dict, rev_edges_dict


def create_data():
    data_dir = "data/derived/"
    train_dataset = GraphDataset(
        config=link_pred_config.dataloader_config,
        edge_path=data_dir + "edges_train.pt",
        graph_path=data_dir + "train_graph.pt",
        train=True,
    )

    return train_dataset


def node_features(data: HeteroData):
    # Basic Testing of node features (if they get mixed up or not)
    user_feature, article_feature = (
        data[Constants.node_user].x,
        data[Constants.node_item].x,
    )
    assert user_feature[0].tolist() == node_1_features
    assert user_feature[0].tolist() == node_1_features
    assert user_feature[1].tolist() == node_2_features

    assert article_feature[0].tolist() == article_1_features
    assert article_feature[0].tolist()[1] == article_1_features[1]
    assert article_feature[1].tolist() == article_2_features
    assert article_feature[1].tolist()[2] == article_2_features[2]
    assert article_feature[2].tolist() == article_3_features
    assert article_feature[2].tolist()[0] == article_3_features[0]


def edge_features(data: HeteroData):

    # Basic Testing of edges
    edges = data[Constants.edge_key]
    assert edges.edge_label_index.shape[0] == 2
    assert edges.edge_label.shape[0] == 2
    assert len(edges.label.shape[0]) == 1

    assert edges.edge_label_index.shape[1] == 2

    # Assertions for reverse edges


def test_integrity():

    data, edges, rev_edges = create_dummy_data()  # create_data()
    node_features(data)
    edge_features(data)
