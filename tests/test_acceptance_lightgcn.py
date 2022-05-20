from config import LightGCNConfig, DataType, PreprocessingConfig
from data.types import UserColumn, ArticleColumn
from run_preprocessing_fashion import preprocess
from run_pipeline_lightgcn import train
from torch_geometric import seed_everything


preprocessing_config = PreprocessingConfig(
    customer_features=[
        UserColumn.PostalCode,
        UserColumn.FN,
        UserColumn.Age,
        UserColumn.ClubMemberStatus,
        UserColumn.FashionNewsFrequency,
        UserColumn.Active,
    ],
    article_features=[
        ArticleColumn.ProductCode,
        ArticleColumn.ProductTypeNo,
        ArticleColumn.GraphicalAppearanceNo,
        ArticleColumn.ColourGroupCode,
    ],
    article_non_categorical_features=[ArticleColumn.ImgEmbedding],
    filter_out_unconnected_nodes=True,
    load_image_embedding=False,
    load_text_embedding=False,
    text_embedding_colname="derived_look",
    data_size=1_000,
    save_to_neo4j=False,
    data_type=DataType.pyg,
    extra_node_type=None,  # ArticleColumn.ProductTypeNo,
    extra_edge_type_label=None,  # "has_type",
)

lightgcn_config = LightGCNConfig(
    epochs=1000,
    k=12,
    hidden_layer_size=32,
    learning_rate=1e-3,
    save_model=False,
    batch_size=128,
    num_iterations=4,
    eval_every=100,
    lr_decay_every=100,
    Lambda=1e-6,
    show_graph=False,
    num_recommendations=256,
)


# def test_lightgcn_pipeline():
#     seed_everything(42)
#     preprocess(preprocessing_config)
#     stats = train(lightgcn_config)
#     assert stats.loss < -0.8
#     assert stats.recall_test > 0.01
#     assert stats.precision_test > 0.0008
