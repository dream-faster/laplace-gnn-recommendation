from torch_geometric.data import HeteroData
from utils.constants import Constants
import torch
from tests.data_generator import save_dummy_data
from torch_geometric import seed_everything
from tests.util import get_first_item_from_dataset

seed_everything(5)
save_dummy_data()

data = get_first_item_from_dataset()


def test_integrity_edges(data: HeteroData = data):

    # Basic Testing of edges if they fit size expectations
    for edge_type in [Constants.edge_key, Constants.rev_edge_key]:
        edges = data[edge_type]
        assert edges.edge_index.shape[0] == 2
        assert edges.edge_index.shape[1] > 0
        assert edges.edge_label_index.shape[0] == 2
        assert edges.edge_label_index.shape[1] > 0
        assert edges.edge_label.shape[0] == 2
        assert edges.edge_label.shape[0] > 0
        assert len(edges.edge_label.shape) == 1
        assert edges.edge_label.shape[0] == edges.edge_label_index.shape[1]


def test_integrity_nodes(data: HeteroData = data):
    user_features, article_features, edge_index, edge_label_index, edge_label = (
        data[Constants.node_user].x,
        data[Constants.node_item].x,
        data[Constants.edge_key].edge_index,
        data[Constants.edge_key].edge_label_index,
        data[Constants.edge_key].edge_label,
    )
    all_touched_users = torch.unique(torch.concat([edge_index[0], edge_label_index[0]]))
    all_touched_articles = torch.unique(
        torch.concat([edge_index[1], edge_label_index[1]])
    )

    print(user_features)
    print(article_features)
    print(edge_index)
    print(edge_label_index)
    print(edge_label)
    print(torch.unique(torch.concat([edge_index[1], edge_label_index[1]])))

    assert torch.equal(
        user_features.type(torch.float), torch.tensor([[0.0, 0.1]]).type(torch.float)
    )
    assert (
        user_features.shape[0] == all_touched_users.shape[0]
    ), "User features are not the same as existing and sampled edges."

    assert (
        article_features.shape[0] == all_touched_articles.shape[0]
    ), "Article features are not the same as existing and sampled edges."

    assert torch.equal(
        article_features.type(torch.float),
        torch.tensor(
            [
                [0.0, 0.1, 0.2, 0.3, 0.4],
                [1.0, 1.1, 1.2, 1.3, 1.4],
                [2.0, 2.1, 2.2, 2.3, 2.4],
            ]
        ).type(torch.float),
    )
