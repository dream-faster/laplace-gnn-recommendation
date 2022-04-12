import torch
from torch_geometric.data import HeteroData, Data
from data.types import FeatureInfo, PipelineConst
from typing import Union, Optional


def __homogenous_features(full_data: Data) -> FeatureInfo:

    data = full_data.x

    return FeatureInfo(
        num_feat=data.shape[1],
        num_cat=torch.max(data, dim=0)[0].tolist(),
        embedding_size=[10] * data.shape[1],
    )


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
    full_data: Union[HeteroData, Data], type: PipelineConst
) -> Optional[Union[tuple[FeatureInfo, FeatureInfo], FeatureInfo]]:
    assert type in [
        PipelineConst.homogenous,
        PipelineConst.heterogenous,
    ], "Invalid type"

    if type == PipelineConst.heterogenous:
        return __heterogenous_features(full_data)

    elif type == PipelineConst.homogenous:
        return __homogenous_features(full_data)

    return None
