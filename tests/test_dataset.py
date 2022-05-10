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
    assert torch.equal(user_feature[0], raw_data["node_features_first"])
    assert torch.equal(user_feature[-1], raw_data["node_features_last"])

    assert torch.equal(
        article_feature[0], raw_data["article_features_first"]
    ), "Article wrong"
    assert torch.equal(
        article_feature[-1], raw_data["article_features_last"]
    ), "Article wrong"


def edge_features(data: HeteroData):

    # Basic Testing of edges
    edges = data[Constants.edge_key]
    assert edges.edge_label_index.shape[0] == 2
    assert edges.edge_label.shape[0] == 2
    assert len(edges.label.shape[0]) == 1

    assert edges.edge_label_index.shape[1] == 2

    # Assertions for reverse edges


def test_integrity():

    # data, raw_data = create_dummy_data()
    data, raw_data = create_data()
    node_features(data, raw_data)
    # edge_features(data, raw_data)


if __name__ == "__main__":
    test_integrity()
