from dataclasses import dataclass
from typing import Optional, Union
from data.types import (
    DataLoaderConfig,
    PreprocessingConfig,
    UserColumn,
    ArticleColumn,
    DataType,
)

embedding_range_dict = {
    "2": 2,
    "10": 3,
    "1000": 6,
    "10000": 20,
    "100000": 30,
    "1000000": 60,
}


@dataclass
class Config:
    epochs: int  # number of training epochs
    hidden_layer_size: int
    k: int  # value of k for recall@k. It is important to set this to a reasonable value!
    # num_layers: int  # number of  layers (i.e., number of hops to consider during propagation)

    # embedding_dim: int  # dimension to use for the customer/article embeddings

    learning_rate: float

    dataloader: bool
    save_model: bool
    dataloader_config: DataLoaderConfig


config = Config(
    epochs=100,
    k=12,
    # num_layers=3,
    hidden_layer_size=128,
    learning_rate=0.01,
    # embedding_dim=64,
    dataloader=True,
    save_model=False,
    dataloader_config=DataLoaderConfig(
        test_split=0.015, val_split=0.015, batch_size=32
    ),
)

only_users_and_articles_nodes = PreprocessingConfig(
    customer_features=[
        UserColumn.PostalCode,
        UserColumn.FN,
        UserColumn.Age,
        UserColumn.ClubMemberStatus,
        UserColumn.FashionNewsFrequency,
        UserColumn.Active,
    ],
    # customer_nodes=[],
    article_features=[
        ArticleColumn.ProductCode,
        ArticleColumn.ProductTypeNo,
        ArticleColumn.GraphicalAppearanceNo,
        ArticleColumn.ColourGroupCode,
    ],
    # article_nodes=[],
    article_non_categorical_features=[ArticleColumn.ImgEmbedding],
    load_image_embedding=False,
    load_text_embedding=False,
    text_embedding_colname="derived_look",
    K=0,
    data_size=100,
    save_to_csv=False,
    data_type=DataType.pyg,
)
