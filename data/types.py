from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Union

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


class DataType(Enum):
    pyg = "pyg"
    dgl = "dgl"


@dataclass
class DataLoaderConfig:
    batch_size: int  # batch size. refers to the # of customers in the batch (each will come with all of its edges)
    val_split: float
    test_split: float


@dataclass
class PreprocessingConfig:
    customer_features: List[UserColumn]
    article_features: List[ArticleColumn]

    article_non_categorical_features: List[ArticleColumn]

    load_image_embedding: bool
    load_text_embedding: bool
    text_embedding_colname: Optional[
        str
    ]  # ["derived_name", "derived_look", "derived_category"]
    K: int
    data_size: Optional[int]
    save_to_csv: Optional[bool]
    data_type: DataType


@dataclass
class FeatureInfo:
    num_feat: int
    num_cat: list[int]
    embedding_size: list[int]
