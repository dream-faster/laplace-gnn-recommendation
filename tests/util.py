from utils.constants import Constants
from torch_geometric.data import HeteroData
from config import Config
from data.dataset import GraphDataset
from torch import Tensor
import pandas as pd
from typing import Tuple, Optional


def get_first_item_from_dataset() -> HeteroData:
    data_dir = "data/derived/"

    config = Config(
        wandb_enabled=False,
        epochs=10,
        k=12,
        num_gnn_layers=2,
        num_linear_layers=2,
        hidden_layer_size=128,
        encoder_layer_output_size=64,
        conv_agg_type="add",
        heterogeneous_prop_agg_type="sum",
        learning_rate=0.01,
        save_model=False,
        test_split=0.1,
        val_split=0.1,
        batch_size=1,  # combination of batch_size with num_neighbors and num_neighbors_it and num_workers determines if data would fit on gpu
        num_neighbors=64,  # -1 takes all neighbors
        num_neighbors_it=2,
        num_workers=1,
        candidate_pool_size=20,
        positive_edges_ratio=0.5,
        negative_edges_ratio=1.0,
        eval_every=1,
        lr_decay_every=1,
        Lambda=1e-6,
        save_every=0.2,  #
        profiler=None,  # Profiler(every=20),
        evaluate_break_at=None,
    )

    train_dataset = GraphDataset(
        config=config,
        users_adj_list=data_dir + "dummy_edges_train.pt",
        graph_path=data_dir + "dummy_graph_train.pt",
        articles_adj_list=data_dir + "dummy_rev_edges_train.pt",
        train=True,
        randomization=False,
    )

    return train_dataset[0]  # type: ignore


def get_raw_sample(data: HeteroData):

    raw_sample = {
        "node_features_first": data[Constants.node_user].x[0],
        "node_features_last": data[Constants.node_user].x[-1],
        "article_features_first": data[Constants.node_item].x[0],
        "article_features_last": data[Constants.node_item].x[-1],
        "edge_index_first": data[Constants.edge_key].edge_index[:, 0],
        "edge_index_last": data[Constants.edge_key].edge_index[:, -1],
    }

    return raw_sample


def get_raw_all(data: HeteroData):

    raw_all = {
        "user_features": data[Constants.node_user].x,
        "article_features": data[Constants.node_item].x,
        "edge_index": data[Constants.edge_key].edge_index,
        "edge_label_index": data[Constants.edge_key].edge_label_index,
        "edge_label": data[Constants.edge_key].edge_label,
    }

    return raw_all


def extract_edges(edges: pd.DataFrame, by: str, get: str) -> dict:
    return edges.groupby(by)[get].apply(list).to_dict()


def deconstruct_heterodata(
    hdata: HeteroData,
) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    return (
        hdata[Constants.node_user].x,
        hdata[Constants.node_item].x,
        hdata[Constants.edge_key].edge_index,
        hdata[Constants.edge_key].edge_label_index
        if hdata[Constants.edge_key].edge_label_index is not None
        else None,
        hdata[Constants.edge_key].edge_label
        if hdata[Constants.edge_key].edge_label is not None
        else None,
    )


def get_edge_dicts(edge_index: Tensor) -> Tuple[dict, dict]:
    edge_index_pd = (
        pd.DataFrame(edge_index.numpy())
        .transpose()
        .rename({0: "user", 1: "article"}, axis=1)
    )
    edges_dict = extract_edges(
        edge_index_pd,
        by="user",
        get="article",
    )
    rev_edges_dict = extract_edges(
        edge_index_pd,
        by="article",
        get="user",
    )

    return edges_dict, rev_edges_dict
