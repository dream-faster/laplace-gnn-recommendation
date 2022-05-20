from config import Config, DataType
from data.types import BasePreprocessingConfig
from run_preprocessing import preprocess
from run_pipeline import run_pipeline
from torch_geometric import seed_everything
from run_download_data import download_movielens

preprocessing_config = BasePreprocessingConfig(
    filter_out_unconnected_nodes=True,
    data_size=1_000,
    save_to_neo4j=False,
    data_type=DataType.pyg,
)

config = Config(
    matchers="movielens",  # "fashion" or "movielens"
    wandb_enabled=False,
    epochs=100,
    k=12,
    num_gnn_layers=2,
    num_linear_layers=2,
    hidden_layer_size=128,
    encoder_layer_output_size=64,
    conv_agg_type="add",
    heterogeneous_prop_agg_type="sum",
    learning_rate=0.01,
    save_model=False,
    batch_size=128,  # combination of batch_size with num_neighbors and n_hop_neighbors and num_workers determines if data would fit on gpu
    num_neighbors=64,  #
    n_hop_neighbors=3,
    num_workers=1,
    candidate_pool_size=20,
    positive_edges_ratio=0.5,
    negative_edges_ratio=3.0,
    eval_every=5,
    save_every=0.2,  #
    profiler=None,  # Profiler(every=20),
    evaluate_break_at=None,
    p_dropout_edges=0.2,  # Currently not being used!
    p_dropout_features=0.3,
    batch_norm=True,
    neo4j=False,
)


def test_pipeline():
    download_movielens()
    seed_everything(42)
    preprocess(preprocessing_config)
    stats = run_pipeline(config)
    assert stats.loss < 0.5
    assert stats.recall_test > 0.0015
    assert stats.precision_test > 0.01
