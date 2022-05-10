from torch_geometric.data import Data, HeteroData
from utils.constants import Constants
import torch
from data.dataset import GraphDataset
from config import link_pred_config
from tests.dummy_data import create_dummy_data
from tests.actual_data import create_data


def node_features(data: HeteroData, raw_data: dict):
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

    data, raw_data = create_dummy_data()  # create_data()
    node_features(data, raw_data)
    edge_features(data, raw_data)
