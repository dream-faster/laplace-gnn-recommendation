import pandas as pd
import torch as t
import numpy as np
from typing import Tuple, Optional
import numpy as np
from utils.constants import Constants


def create_data_pyg(
    customers: t.Tensor,
    articles: t.Tensor,
    extra_nodes: Optional[t.Tensor],
    extra_node_name: Optional[str],
    transactions_to_customer_id: np.ndarray,
    transactions_to_article_id: np.ndarray,
    extra_edges_from_article_id: Optional[np.ndarray],
    extra_edges_to_extra_node_id: Optional[np.ndarray],
    extra_edge_type_label: Optional[str],
):

    from torch_geometric.data import HeteroData

    data = HeteroData()
    data[Constants.node_user].x = customers
    data[Constants.node_item].x = articles
    if extra_nodes is not None:
        data[extra_node_name].x = extra_nodes

    data[Constants.edge_key].edge_index = t.as_tensor(
        (transactions_to_customer_id, transactions_to_article_id),
        dtype=t.long,
    )
    if extra_edge_type_label is not None:
        data[
            (extra_node_name, extra_edge_type_label, Constants.node_item)
        ].edge_index = t.as_tensor(
            (extra_edges_from_article_id, extra_edges_to_extra_node_id),
            dtype=t.long,
        )
    return data


def create_data_dgl(
    customers: t.Tensor,
    articles: t.Tensor,
    transactions_to_customer_id: np.ndarray,
    transactions_to_article_id: np.ndarray,
):

    import dgl

    data = dgl.heterograph(
        {
            Constants.edge_key: (
                t.as_tensor(transactions_to_customer_id, dtype=t.long),
                t.as_tensor(transactions_to_article_id, dtype=t.long),
            ),
            Constants.rev_edge_key: (
                t.as_tensor(transactions_to_article_id, dtype=t.long),
                t.as_tensor(transactions_to_customer_id, dtype=t.long),
            ),
        },
        num_nodes_dict={
            Constants.node_user: customers.shape[0],
            Constants.node_item: articles.shape[0],
        },
    )
    data.nodes[Constants.node_user].data["features"] = customers
    data.nodes[Constants.node_item].data["features"] = articles
    return data


def create_ids_and_maps(
    df: pd.DataFrame, column: str, start: int
) -> Tuple[pd.DataFrame, dict, dict]:
    df.reset_index(inplace=True)
    df.index += start
    mapping_forward = df[column].to_dict()
    mapping_reverse = {v: k for k, v in mapping_forward.items()}
    df["index"] = df.index
    return df, mapping_forward, mapping_reverse


def extract_edges(transactions: pd.DataFrame) -> dict:
    return transactions.groupby("customer_id")["article_id"].apply(list).to_dict()


def extract_reverse_edges(transactions: pd.DataFrame) -> dict:
    return transactions.groupby("article_id")["customer_id"].apply(list).to_dict()
