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
    num_layers: int  # number of  layers (i.e., number of hops to consider during propagation)
    learning_rate: float
    save_model: bool
    dataloader_config: DataLoaderConfig
    eval_every: int  # (LightGCN) evaluation to run every n epoch
    lr_decay_every: int  # (LightGCN) lr decay to run every n epoch
    Lambda: float  # (LightGCN)
    save_every: int  # How often the model should be saved


link_pred_config = Config(
    epochs=10,
    k=12,
    num_layers=3,
    hidden_layer_size=128,
    learning_rate=0.01,
    save_model=False,
    dataloader_config=DataLoaderConfig(
        test_split=0.1,
        val_split=0.1,
        batch_size=12,  # combination of batch_size with num_neighbors and num_neighbors_it and num_workers determines if data would fit on gpu
        num_neighbors=64,  # -1 takes all neighbors
        num_neighbors_it=2,
        num_workers=1,
        candidate_pool_size=20,
    ),
    eval_every=1,
    lr_decay_every=1,
    Lambda=1e-6,
    save_every=2,
)


lightgcn_config = Config(
    epochs=100,
    k=12,
    num_layers=3,  # Number of LightGCN steps
    hidden_layer_size=32,
    learning_rate=1e-3,
    save_model=False,
    dataloader_config=DataLoaderConfig(
        test_split=0.1,
        val_split=0.1,
        batch_size=128,
        num_neighbors=0,  # IGNORE for LightGCN
        num_neighbors_it=0,  # IGNORE for LightGCN
        num_workers=1,
        candidate_pool_size=None,
    ),
    eval_every=100,
    lr_decay_every=100,
    Lambda=1e-6,
    save_every=1,
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
    filter_out_unconnected_nodes=True,
    load_image_embedding=False,
    load_text_embedding=False,
    text_embedding_colname="derived_look",
    K=0,
    data_size=1000,
    save_to_csv=False,
    data_type=DataType.pyg,
)
