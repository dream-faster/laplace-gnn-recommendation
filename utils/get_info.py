import torch
from torch_geometric.data import HeteroData, Data
from data.types import FeatureInfo
from typing import Union, Tuple
from config import embedding_range_dict


def __embedding_size_selector(max_category: int):
    for key, value in embedding_range_dict.items():
        if max_category <= int(key):
            return value
    return embedding_range_dict["10000"]


def __heterogenous_features(full_data: HeteroData) -> Tuple[FeatureInfo, FeatureInfo]:
    customer_features = full_data.x_dict["customer"]
    article_features = full_data.x_dict["article"]

    customer_num_cat = torch.max(customer_features, dim=0)[0].tolist()
    article_num_cat = torch.max(article_features, dim=0)[0].tolist()

    customer_feat_info, article_feat_info = FeatureInfo(
        num_feat=customer_features.shape[1],
        num_cat=customer_num_cat,
        embedding_size=[
            __embedding_size_selector(max_cat) for max_cat in customer_num_cat
        ],
    ), FeatureInfo(
        num_feat=article_features.shape[1],
        num_cat=article_num_cat,
        embedding_size=[
            __embedding_size_selector(max_cat) for max_cat in article_num_cat
        ],
    )

    return (customer_feat_info, article_feat_info)


def get_feature_info(
    full_data: Union[HeteroData, Data]
) -> Tuple[FeatureInfo, FeatureInfo]:

    return __heterogenous_features(full_data)
