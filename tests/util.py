from utils.constants import Constants
from torch_geometric.data import HeteroData

def get_raw_sample(data: HeteroData):

    raw_sample = {
        "node_features_first": data[Constants.node_user].x.tolist()[0],
        "node_features_last": data[Constants.node_user].x.tolist()[-1],
        "article_features_first": data[Constants.node_item].x.tolist()[0],
        "article_features_last": data[Constants.node_item].x.tolist()[-1],
        "edge_index_first": data[Constants.edge_key].edge_index.tolist()[:, 0],
        "edge_index_last": data[Constants.edge_key].edge_index.tolist()[:, -1],
        "edge_label_index_first": data[Constants.edge_key].edge_label_index.tolist()[
            :, 0
        ],
        "edge_label_index_last": data[Constants.edge_key].edge_label_index.tolist()[
            :, -1
        ],
        "edge_label_first": data[Constants.edge_key].edge_label.tolist()[0],
        "edge_label_last": data[Constants.edge_key].edge_label.tolist()[-1],
    }

    return raw_sample