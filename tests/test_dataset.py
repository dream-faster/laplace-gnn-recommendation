from email.contentmanager import raw_data_manager
from torch_geometric.data import Data, HeteroData
from utils.constants import Constants
import torch
from data.dataset import GraphDataset
from config import link_pred_config
from tests.dummy_data import create_dummy_data, get_raw_data
from tests.actual_data import create_data
from torch_geometric import seed_everything

seed_everything(5)
data = create_data()


def edge_features(data: HeteroData):

    # Basic Testing of edges if they fit size expectations
    edges = data[Constants.edge_key]
    assert edges.edge_index.shape[0] == 2
    assert edges.edge_index.shape[1] > 0
    assert edges.edge_label_index.shape[0] == 2
    assert edges.edge_label_index.shape[1] > 0
    assert edges.edge_label.shape[0] == 2
    assert edges.edge_label.shape[0] > 0
    assert len(edges.edge_label.shape) == 1
    assert edges.edge_label.shape[0] == edges.edge_label_index.shape[1]

    # Assertions for reverse edges


def node_features(data: HeteroData):
    user_features, article_features = (
        data[Constants.node_user].x,
        data[Constants.node_item].x,
    )
    # print(data)
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
        torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [1.1, 1.2, 1.3, 1.4],
                [2.1, 2.2, 2.3, 2.4],
            ]
        ).type(torch.float),
    )


def test_integrity_nodes():
    node_features(data)


def test_integrity_edges():
    edge_features(data)


if __name__ == "__main__":
    test_integrity_nodes()
