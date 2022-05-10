from email.contentmanager import raw_data_manager
from torch_geometric.data import Data, HeteroData
from utils.constants import Constants
import torch
from data.dataset import GraphDataset
from config import link_pred_config
from tests.dummy_data import create_dummy_data, get_raw_data
from tests.actual_data import create_data


def node_features(data: HeteroData, raw_data: dict):
    # Basic Testing of node features (if they get mixed up or not)
    user_feature, article_feature = (
        data[Constants.node_user].x,
        data[Constants.node_item].x,
    )
    assert torch.equal(user_feature[0], raw_data["node_features_first"])
    # assert torch.equal(user_feature[-1], raw_data["node_features_last"])

    # assert torch.equal(
    #     article_feature[0], raw_data["article_features_first"]
    # ), "Article wrong"
    # assert torch.equal(
    #     article_feature[-1], raw_data["article_features_last"]
    # ), "Article wrong"


def edge_features(data: HeteroData):

    # Basic Testing of edges
    edges = data[Constants.edge_key]
    assert edges.edge_label_index.shape[0] == 2
    assert edges.edge_label.shape[0] == 2
    assert len(edges.label.shape[0]) == 1

    assert edges.edge_label_index.shape[1] == 2

    # Assertions for reverse edges


def node_features_manual(data: HeteroData, raw_data: dict):
    # positive_node_edges_random_id = torch.tensor([0])
    # # Taken from a run through the dataset
    # negative_node_edges_random_id = torch.tensor(
    #     [0]
    # )  # Taken from a run through the dataset

    user_features, article_features = (
        data[Constants.node_user].x,
        data[Constants.node_item].x,
    )
    print(data)
    print(data[Constants.edge_key].edge_index)
    print(data[Constants.edge_key].edge_label_index)
    print(data[Constants.edge_key].edge_label)
    print(user_features)
    print(article_features)

    assert torch.equal(
        user_features.type(torch.float), torch.tensor([[0.3, 0.2]]).type(torch.float)
    )
    assert torch.equal(
        article_features.type(torch.float),
        torch.tensor([[0.2, 0.3, 0.4, 0.5]]).type(torch.float),
    )


def test_integrity():
    data = create_data()
    _, _, raw_all = create_dummy_data()
    node_features_manual(data, raw_all)
    # edge_features(data, raw_data)


# def test_integrity_dummy():
#     data, raw_data = create_dummy_data()
#     node_features(data, raw_data)
#     # edge_features(data, raw_data)


if __name__ == "__main__":
    test_integrity()
