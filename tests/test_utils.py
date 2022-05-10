from utils.constants import Constants
from torch_geometric.data import HeteroData


def get_raw_sample(data: HeteroData):

    raw_sample = {
        "node_features_first": data[Constants.node_user].x[0],
        "node_features_last": data[Constants.node_user].x[-1],
        "article_features_first": data[Constants.node_item].x[0],
        "article_features_last": data[Constants.node_item].x[-1],
        "edge_index_first": data[Constants.edge_key].edge_index[:, 0],
        "edge_index_last": data[Constants.edge_key].edge_index[:, -1],
        # "edge_label_index_first": data[Constants.edge_key].edge_label_index[:, 0],
        # "edge_label_index_last": data[Constants.edge_key].edge_label_index[:, -1],
        # "edge_label_first": data[Constants.edge_key].edge_label[0],
        # "edge_label_last": data[Constants.edge_key].edge_label[-1],
    }

    return raw_sample


def get_raw_all(data: HeteroData):

    raw_all = {
        "user_features": data[Constants.node_user].x,
        "article_features": data[Constants.node_item].x,
        "edge_index": data[Constants.edge_key].edge_index,
        "edge_label_index": data[Constants.edge_key].edge_label_index,
        "edge_label": data[Constants.edge_key].edge_label,
    }

    return raw_all
