import torch
from torch_geometric.data import HeteroData
from data.types import FeatureInfo
from typing import Union


def __homogenous_features(full_data: HeteroData) -> FeatureInfo:
    # FeatureInfo(
    #     num_feat=article_features.shape[1],
    #     num_cat=article_num_cat.tolist(),
    #     embedding_size=[10] * article_features.shape[1],
    # )
    pass


def __heterogenous_features(full_data: HeteroData) -> tuple[FeatureInfo, FeatureInfo]:
    customer_features = full_data.x_dict["customer"]
    article_features = full_data.x_dict["article"]

    customer_num_cat, _ = torch.max(customer_features, dim=0)
    article_num_cat, _ = torch.max(article_features, dim=0)

    customer_feat_info, article_feat_info = FeatureInfo(
        num_feat=customer_features.shape[1],
        num_cat=customer_num_cat.tolist(),
        embedding_size=[10] * customer_features.shape[1],
    ), FeatureInfo(
        num_feat=article_features.shape[1],
        num_cat=article_num_cat.tolist(),
        embedding_size=[10] * article_features.shape[1],
    )

    return (customer_feat_info, article_feat_info)


def get_feature_info(
    full_data: HeteroData, type: str
) -> Union[tuple[FeatureInfo, FeatureInfo], FeatureInfo]:
    if type == "heterogenous":
        return __heterogenous_features(full_data)

    if type == "homogenous":
        return __heterogenous_features(full_data)
