import torch as t
import numpy as np
from torch import Tensor
from typing import List, Tuple
from .metrics import RecallPrecision_ATk, NDCGatK_r


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
def get_metrics_universal(
    model_output,
    edge_index: Tensor,
    edge_label_index: Tensor,
    exclude_edge_indices: List[Tensor],
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

    if len(model_output.shape) < 2:
        ratings = model_output.unsqueeze(0)
    else:
        ratings = model_output

    for exclude_edge_index in exclude_edge_indices:
        # gets all the positive items for each user from the edge index
        user_pos_items = get_user_positive_items(exclude_edge_index)
        # get coordinates of all edges to exclude
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        # set ratings of excluded edges to large negative value
        ratings[exclude_users, exclude_items] = -(1 << 10)

    # get the top k recommended items for each user
    _, top_K_items = t.topk(ratings, k=k)

    # get all unique users in evaluated split
    users = edge_label_index[0].unique()

    test_user_pos_items = get_user_positive_items(edge_index)

    # convert test user pos items dictionary into a list
    test_user_pos_items_list = [test_user_pos_items[user.item()] for user in users]

    # determine the correctness of topk predictions
    r = []
    for i, user in enumerate(users):
        ground_truth_items = test_user_pos_items[user.item()]
        label = [x in ground_truth_items for x in top_K_items[i]]
        r.append(label)
    r = t.Tensor(np.array(r).astype("float"))

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)

    return recall, precision, ndcg
