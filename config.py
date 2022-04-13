from dataclasses import dataclass
from typing import Optional, Union
from data.types import (
    GraphType,
    PreprocessingConfig,
    UserColumn,
    ArticleColumn,
    DataType,
)


@dataclass
class Config:
    epochs: int  # number of training epochs
    k: int  # value of k for recall@k. It is important to set this to a reasonable value!
    num_layers: int  # number of LightGCN layers (i.e., number of hops to consider during propagation)
    batch_size: int  # batch size. refers to the # of customers in the batch (each will come with all of its edges)
    embedding_dim: int  # dimension to use for the customer/article embeddings
    save_emb_dir: Optional[
        str
    ]  # path to save multi-scale embeddings during test(). If None, will not save any embeddings
    type: GraphType  # type of graph we use
    dataloader: bool


config = Config(
    epochs=1500,
    k=12,
    num_layers=3,
    batch_size=32,
    embedding_dim=64,
    save_emb_dir=None,
    type=GraphType.heterogenous,
    dataloader=True,
)

only_users_and_articles_nodes = PreprocessingConfig(
    type=GraphType.heterogenous,
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
    data_size=10000,
    save_to_csv=False,
    data_type=DataType.pyg,
)
