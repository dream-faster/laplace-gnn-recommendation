import torch as t
import numpy as np
from torch import Tensor
from typing import List, Tuple
from .metrics import RecallPrecision_ATk, NDCGatK_r
from .edges import create_adj_dict, create_adj_list

# helper function to get N_u
def get_user_positive_items(edge_index: Tensor) -> dict:
    """Generates dictionary of positive items for each user

    Args:
        edge_index (t.Tensor): 2 by N list of edges

    Returns:
        dict: dictionary of positive items for each user
    """
    user_pos_items: dict = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items


# wrapper function to get evaluation metrics
def get_metrics_encoder_decoder(
    model_output,
    edge_index: Tensor,
    edge_label_index: Tensor,
    k: int,
) -> Tuple[float, float, float]:
    """Computes the evaluation metrics: recall, precision, and ndcg @ k

    Args:
        model (LighGCN): ANY model
        edge_index (t.Tensor): 2 by N list of edges for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on

    Returns:
        tuple: recall @ k, precision @ k, ndcg @ k
    """

    # Ratings expects a tensor of ratings for multiple users
    edge_index = edge_index.detach().to("cpu")
    edge_label_index = edge_label_index.detach().to("cpu")
    model_output = model_output.detach().to("cpu")

    if len(model_output.shape) < 2:
        ratings = model_output.unsqueeze(0)
    else:
        ratings = model_output

    # get the top k recommended items for each user
    _, top_K_items = t.topk(ratings, k=k)

    # get all unique users in evaluated split
    users = edge_label_index[0].unique(sorted=True)
    test_user_pos_items = create_adj_list(edge_index, users)

    # determine the correctness of topk predictions
    r = t.stack(
        [t.isin(top_K_items[i], test_user_pos_items[i]) for i, _ in enumerate(users)]
    )

    recall, precision = RecallPrecision_ATk(test_user_pos_items, r, k)
    ndcg = NDCGatK_r(test_user_pos_items, r, k)

    return recall, precision, ndcg
