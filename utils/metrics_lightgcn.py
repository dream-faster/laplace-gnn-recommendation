import torch as t
import numpy as np
from torch import Tensor
from typing import List, Tuple, Optional
from utils.tensor import difference_1d
from .metrics import RecallPrecision_ATk, NDCGatK_r


def bpr_loss(
    users_emb_final: Tensor,
    users_emb_0: Tensor,
    pos_items_emb_final: Tensor,
    pos_items_emb_0: Tensor,
    neg_items_emb_final: Tensor,
    neg_items_emb_0: Tensor,
    lambda_val: float,
) -> Tensor:
    """Bayesian Personalized Ranking Loss as described in https://arxiv.org/abs/1205.2618

    Args:
        users_emb_final (t.Tensor): e_u_k
        users_emb_0 (t.Tensor): e_u_0
        pos_items_emb_final (t.Tensor): positive e_i_k
        pos_items_emb_0 (t.Tensor): positive e_i_0
        neg_items_emb_final (t.Tensor): negative e_i_k
        neg_items_emb_0 (t.Tensor): negative e_i_0
        lambda_val (float): lambda value for regularization loss term

    Returns:
        t.Tensor: scalar bpr loss value
    """
    reg_loss = lambda_val * (
        users_emb_0.norm(2).pow(2)
        + pos_items_emb_0.norm(2).pow(2)
        + neg_items_emb_0.norm(2).pow(2)
    )  # L2 loss

    pos_scores = t.mul(users_emb_final, pos_items_emb_final)
    pos_scores = t.sum(pos_scores, dim=-1)  # predicted scores of positive samples
    neg_scores = t.mul(users_emb_final, neg_items_emb_final)
    neg_scores = t.sum(neg_scores, dim=-1)  # predicted scores of negative samples

    loss = -t.mean(t.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss

    return loss


def create_adj_dict(edge_index: Tensor, from_nodes: Optional[Tensor] = None) -> dict:
    """Generates dictionary of items for each user

    Args:
        edge_index (t.Tensor): 2 by N list of edges

    Returns:
        dict: dictionary of items for each user
    """
    users = edge_index[0].unique(sorted=True) if from_nodes is None else from_nodes
    items_per_user: dict = {}
    for user in users:
        items_per_user[user.item()] = edge_index[1][edge_index[0] == user]
    return items_per_user


def create_adj_list(
    edge_index: Tensor, from_nodes: Optional[Tensor] = None
) -> List[Tensor]:
    """Generates list of items for each user

    Args:
        edge_index (t.Tensor): 2 by N list of edges

    Returns:
        tensor: List[Tensor] of items for each user
    """
    users = edge_index[0].unique(sorted=True) if from_nodes is None else from_nodes
    return [edge_index[1][edge_index[0] == user] for user in users]


def get_metrics_lightgcn(
    model, edge_index: Tensor, exclude_edge_indices: List[Tensor], k: int
) -> Tuple[float, float, float]:
    """Computes the evaluation metrics: recall, precision, and ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (t.Tensor): 2 by N list of edges for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on

    Returns:
        tuple: recall @ k, precision @ k, ndcg @ k
    """
    user_embedding = model.users_emb.weight.detach().to("cpu")
    item_embedding = model.items_emb.weight.detach().to("cpu")

    excluded_edges_per_user = create_adj_dict(t.cat(exclude_edge_indices, dim=1))

    # get all unique users in evaluated split
    users = edge_index[0].unique()

    # get the top k recommended items for each user
    top_K_items = {}
    for user in users:
        top_K_items[user.item()] = make_predictions_for_user(
            user_embedding, item_embedding, user.item(), excluded_edges_per_user, k
        )

    test_user_pos_items = create_adj_dict(edge_index, from_nodes=users)
    test_user_pos_items_list = [test_user_pos_items[user.item()] for user in users]

    # determine the correctness of topk predictions
    r = t.stack(
        [
            t.isin(top_K_items[user.item()], test_user_pos_items[user.item()])
            for user in users
        ]
    )

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)

    return recall, precision, ndcg


def make_predictions_for_user(
    user_embeddings: t.Tensor,
    article_embeddings: t.Tensor,
    user_id: int,
    positive_items_for_user: dict,
    num_recommendations: int,
) -> t.Tensor:
    articles_to_ignore = (
        positive_items_for_user[user_id]
        if user_id in positive_items_for_user
        else t.tensor([])
    )
    scores = user_embeddings[user_id] @ article_embeddings.T

    _, indices = t.topk(scores, k=num_recommendations + len(articles_to_ignore))
    # remove positive items, we don't want to recommend them
    indices = difference_1d(indices, articles_to_ignore, assume_unique=True)
    return indices[:num_recommendations]
