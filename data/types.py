from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, List

ArticleIdMap = dict
CustomerIdMap = dict


class UserColumn(Enum):
    PostalCode = "postal_code"
    FN = "FN"
    Age = "age"
    ClubMemberStatus = "club_member_status"
    FashionNewsFrequency = "fashion_news_frequency"
    Active = "Active"


class ArticleColumn(Enum):
    ProductCode = "product_code"
    ProductTypeNo = "product_type_no"
    GraphicalAppearanceNo = "graphical_appearance_no"
    ColourGroupCode = "colour_group_code"
    AvgPrice = "avg_price"
    ImgEmbedding = "img_embedding"


@dataclass
class DataLoaderConfig:
    batch_size: int
    val_split: float
    test_split: float


@dataclass
class PreprocessingConfig:
    type: Literal['heterogeneous', 'homogeneous']

    customer_features: List[UserColumn]
    # customer_nodes: List[UserColumn]

    article_features: List[ArticleColumn]
    # article_nodes: List[ArticleColumn]

    article_non_categorical_features: List[ArticleColumn]

    load_image_embedding: bool
    K: int
    data_size: Optional[int]
    save_to_csv: Optional[bool]


@dataclass
class FeatureInfo:
    num_feat: int
    num_cat: list[int]
    embedding_size: list[int]
