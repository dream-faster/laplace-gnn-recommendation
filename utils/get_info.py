import torch as t
from torch import Tensor
from torch_geometric.data import HeteroData, Data
from data.types import FeatureInfo
from typing import Union, Tuple
from config import embedding_range_dict
from utils.constants import Constants


def __embedding_size_selector(max_category: int):
    for key, value in embedding_range_dict.items():
        if max_category <= int(key):
            return value
    return embedding_range_dict["10000"]


def __heterogenous_features(data: HeteroData) -> dict[FeatureInfo]:
    node_types, _ = data.metadata()
    feature_info_dict = dict()

    for node_type in node_types:
        features = data.x_dict[node_type]
        num_cat = t.max(features, dim=0)[0].tolist()
        feat_info = FeatureInfo(
            num_feat=features.shape[1],
            num_cat=num_cat,
            embedding_size=[__embedding_size_selector(max_cat) for max_cat in num_cat],
        )
        feature_info_dict[node_type] = feat_info

    return feature_info_dict


def get_feature_info(full_data: Union[HeteroData, Data]) -> dict[FeatureInfo]:

    return __heterogenous_features(full_data)


def select_properties(
    data: Union[HeteroData, Data]
) -> Tuple[dict, dict, Tensor, Tensor]:

    return (
        data.x_dict,
        data.edge_index_dict,
        data[Constants.edge_key].edge_label_index,
        data[Constants.edge_key].edge_label.float(),
    )
