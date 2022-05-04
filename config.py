from dataclasses import dataclass
from typing import Optional, Union
from data.types import (
    DataLoaderConfig,
    PreprocessingConfig,
    UserColumn,
    ArticleColumn,
    DataType,
)
from utils.profiling import Profiler

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
    save_every: float  # How often the model should be saved, Ratio of epochs (eg.: 0.2 * epoch_num)
    positive_edges_ratio: float  # ratio of positive edges that we sample for edge_label_index, eg.: 0.5 means we take the half of the avilable edges from that user, the result won't be less than 1 (We will always sample at least one positive edge)
    negative_edge_ratio: float  # How many negative edges to sample based on the positive ones, eg.: 10 means we take 10*sampled_positive_edges
    profiler: Optional[Profiler] = None
    evaluate_break_at: Optional[
        int
    ] = None  # Eval and Test should break after this many iterations (not epochs!) None runs whole test and val

    def print(self):
        print("\x1b[1;32;47m")
        print("Configuration is:")
        for key, value in vars(self).items():
            print("\x1b[1;37;47m" + f"{key:>20}: " + "\x1b[0;32;47m" + f"{value}")
        print("\x1b[0m")


link_pred_config = Config(
    epochs=100,
    k=12,
    num_layers=1,
    hidden_layer_size=128,
    learning_rate=0.01,
    save_model=False,
    dataloader_config=DataLoaderConfig(
        test_split=0.1,
        val_split=0.1,
        batch_size=128,  # combination of batch_size with num_neighbors and num_neighbors_it and num_workers determines if data would fit on gpu
        num_neighbors=64,  # -1 takes all neighbors
        num_neighbors_it=2,
        num_workers=1,
        candidate_pool_size=20,
    ),
    eval_every=1,
    lr_decay_every=1,
    Lambda=1e-6,
    save_every=0.2,  #
    profiler=None,  # Profiler(every=20),
    evaluate_break_at=None,
    positive_edges_ratio=0.5,
    negative_edge_ratio=10.0,
)


lightgcn_config = Config(
    epochs=1000,
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
        candidate_pool_size=None,  # IGNORE for LightGCN
    ),
    eval_every=100,
    lr_decay_every=100,
    Lambda=1e-6,
    save_every=0.2,
    profiler=None,  # IGNORE for LightGCN
    evaluate_break_at=None,  # IGNORE for LightGCN
    positive_edges_ratio=1.0,  # IGNORE for LightGCN
    negative_edge_ratio=1.0,  # IGNORE for LightGCN
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
    data_size=10000,
    save_to_csv=False,
    data_type=DataType.pyg,
)
