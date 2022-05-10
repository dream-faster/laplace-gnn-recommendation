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
class PreprocessingConfig:
    customer_features: List[UserColumn]
    article_features: List[ArticleColumn]

    article_non_categorical_features: List[ArticleColumn]
    filter_out_unconnected_nodes: bool

    load_image_embedding: bool
    load_text_embedding: bool
    text_embedding_colname: Optional[
        str
    ]  # ["derived_name", "derived_look", "derived_category"]
    K: int
    data_size: Optional[int]
    save_to_csv: Optional[bool]
    data_type: DataType

    def print(self):
        print("\x1b[1;32;47m")
        print("Configuration is:")
        for key, value in vars(self).items():
            print("\x1b[1;37;47m" + f"{key:>20}: " + "\x1b[0;32;47m" + f"{value}")
        print("\x1b[0m")


@dataclass
class FeatureInfo:
    num_feat: int
    num_cat: List[int]
    embedding_size: List[int]
