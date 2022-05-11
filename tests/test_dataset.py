from torch_geometric.data import HeteroData
from utils.constants import Constants
import torch as t
from tests.data_generator import create_dummy_data, create_subgraph_comparison
from torch_geometric import seed_everything
from tests.util import get_first_item_from_dataset

seed_everything(5)
original_data = create_dummy_data(save=True)

data_from_dataset = get_first_item_from_dataset()
data_comparison = create_subgraph_comparison(n_hop=2)


def test_integrity_edges(data: HeteroData = data_from_dataset, data_comp: HeteroData = data_comparison):

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
    
    for key, val in vars(data).items():
        assert vars(data_comp)[key] == val

def test_integrity_nodes(data: HeteroData = data_from_dataset):
    user_features, article_features, edge_index, edge_label_index, edge_label = (
        data[Constants.node_user].x,
        data[Constants.node_item].x,
        data[Constants.edge_key].edge_index,
        data[Constants.edge_key].edge_label_index,
        data[Constants.edge_key].edge_label,
    )
    all_touched_users = t.unique(t.concat([edge_index[0], edge_label_index[0]]))
    all_touched_articles = t.unique(t.concat([edge_index[1], edge_label_index[1]]))

    print(user_features)
    print(article_features)
    print(edge_index)
    print(edge_label_index)
    print(edge_label)

    assert t.equal(user_features, t.tensor([[0.0, 0.1]]))
    assert (
        user_features.shape[0] == all_touched_users.shape[0]
    ), "User features are not the same as existing and sampled edges."

    assert (
        article_features.shape[0] == all_touched_articles.shape[0]
    ), "Article features are not the same as existing and sampled edges."

    assert t.equal(
        article_features,
    )
