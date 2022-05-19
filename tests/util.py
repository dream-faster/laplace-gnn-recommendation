from utils.constants import Constants
from torch_geometric.data import HeteroData
from config import Config
from data.dataset import GraphDataset
from data.dataset_neo import GraphDataset as GraphDatasetNeo
from torch import Tensor
import torch as t
import pandas as pd
from typing import Tuple, Optional
from utils.types import NodeFeatures, ArticleFeatures, AllEdges, SampledEdges, Labels
from run_preprocessing_fashion import save_to_neo4j


def get_first_item_from_dataset(graph_database: bool) -> HeteroData:
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
        batch_size=1,  # combination of batch_size with num_neighbors and n_hop_neighbors and num_workers determines if data would fit on gpu
        num_neighbors=64,  # -1 takes all neighbors
        n_hop_neighbors=2,
        num_workers=1,
        candidate_pool_size=20,
        positive_edges_ratio=0.5,
        negative_edges_ratio=1.0,
        eval_every=1,
        save_every=0.2,  #
        profiler=None,  # Profiler(every=20),
        evaluate_break_at=None,
        p_dropout_edges=0.0,
        p_dropout_features=0.0,
        batch_norm=True,
        matchers="fashion",
    )

    if graph_database:
        train_dataset = GraphDatasetNeo(
            config=config,
            users_adj_list=data_dir + "dummy_edges_train.pt",
            graph_path=data_dir + "dummy_graph_train.pt",
            articles_adj_list=data_dir + "dummy_rev_edges_train.pt",
            train=True,
            randomization=False,
            split_type="train",
        )

    else:
        train_dataset = GraphDataset(
            config=config,
            users_adj_list=data_dir + "dummy_edges_train.pt",
            graph_path=data_dir + "dummy_graph_train.pt",
            articles_adj_list=data_dir + "dummy_rev_edges_train.pt",
            train=True,
            randomization=False,
        )

    return train_dataset[0]  # type: ignore


def construct_heterodata(
    user_features: NodeFeatures,
    article_features: ArticleFeatures,
    edge_index: AllEdges,
    edge_label_index: Optional[SampledEdges],
    edge_label: Optional[Labels],
) -> HeteroData:

    """Create Data"""
    data = HeteroData()
    data[Constants.node_user].x = user_features
    data[Constants.node_item].x = article_features

    # Add original directional edges
    data[Constants.edge_key].edge_index = edge_index
    if type(edge_label_index) is SampledEdges:
        data[Constants.edge_key].edge_label_index = edge_label_index
    if type(edge_label) is Labels:
        data[Constants.edge_key].edge_label = edge_label

    # Add reverse edges
    reverse_key = t.LongTensor([1, 0])
    data[Constants.rev_edge_key].edge_index = edge_index[reverse_key]
    if type(edge_label_index) is SampledEdges:
        data[Constants.rev_edge_key].edge_label_index = edge_label_index[reverse_key]
    if type(edge_label) is Labels:
        data[Constants.rev_edge_key].edge_label = edge_label

    return data


def deconstruct_heterodata(
    hdata: HeteroData,
) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    return (
        hdata[Constants.node_user].x,
        hdata[Constants.node_item].x,
        hdata[Constants.edge_key].edge_index,
        hdata[Constants.edge_key].edge_label_index
        if hasattr(hdata[Constants.edge_key], "edge_label_index")
        else None,
        hdata[Constants.edge_key].edge_label
        if hasattr(hdata[Constants.edge_key], "edge_label")
        else None,
    )


def get_edge_dicts(edge_index: Tensor) -> Tuple[dict, dict]:
    edge_index_pd = (
        pd.DataFrame(edge_index.numpy())
        .transpose()
        .rename({0: "user", 1: "article"}, axis=1)
    )
    edges_dict = __extract_edges(
        edge_index_pd,
        by="user",
        get="article",
    )
    rev_edges_dict = __extract_edges(
        edge_index_pd,
        by="article",
        get="user",
    )

    return edges_dict, rev_edges_dict


def __extract_edges(edges: pd.DataFrame, by: str, get: str) -> dict:
    return edges.groupby(by)[get].apply(list).to_dict()


def preprocess_and_load_to_neo4j(
    original_data: HeteroData,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customers = pd.DataFrame(original_data[Constants.node_user].x)
    customers.rename(
        columns={
            i: column_name
            for i, column_name in enumerate(
                [
                    "postal_code",
                    "FN",
                    "age",
                    "club_member_status",
                    "fashion_news_frequency",
                    "Active",
                ]
            )
        },
        inplace=True,
    )
    customers[":ID(Customer)"] = customers.index
    customers[":LABEL"] = "Customer"
    customers["_id"] = customers[":ID(Customer)"]

    articles = pd.DataFrame(original_data[Constants.node_item].x)
    articles.rename(
        columns={
            i: column_name
            for i, column_name in enumerate(
                [
                    "product_code",
                    "product_type_no",
                    "graphical_appearance_no",
                    "colour_group_code",
                ]
            )
        },
        inplace=True,
    )
    articles[":ID(Article)"] = articles.index
    articles[":LABEL"] = "Article"
    articles["_id"] = articles[":ID(Article)"]

    transactions = pd.DataFrame(
        {
            ":START_ID(Customer)": original_data[Constants.edge_key]
            .edge_index[0]
            .tolist(),
            ":END_ID(Article)": original_data[Constants.edge_key]
            .edge_index[1]
            .tolist(),
            ":TYPE": "BUYS",
            "train_mask": 1.0,
            "val_mask": 0.0,
            "test_mask": 0.0,
        }
    )

    save_to_neo4j(customers, articles, transactions)
