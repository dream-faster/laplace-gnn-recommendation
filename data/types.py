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
    num_neighbors: int  # sample n neighbors for each node for num_neighbors_it iterations
    num_neighbors_it: int
    num_workers: int  # number of workers to use for data loading
    candidate_pool_size: int  # How many precalculated candidates we should give over
    positive_edges_ratio: float  # Ratio of positive edges that we sample for edge_label_index, eg.: 0.5 means we take the half of the avilable edges from that user, the result won't be less than 1 (We will always sample at least one positive edge)
    negative_edges_ratio: float  # How many negative edges to sample based on the positive ones, eg.: 10 means we take 10*sampled_positive_edges


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
