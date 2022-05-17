import torch as t
import numpy as np
from torch import Tensor
from typing import List, Tuple


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


# helper function to get N_u
def create_edges_dict_indexed_by_user(edge_index: Tensor) -> dict:
    """Generates dictionary of items for each user

    Args:
        edge_index (t.Tensor): 2 by N list of edges

    Returns:
        dict: dictionary of items for each user
    """
    users = edge_index[0].unique(sorted=True)
    items_per_user: dict = {}
    for user in users:
        items_per_user[user.item()] = edge_index[1][edge_index[0] == user]
    return items_per_user


# computes recall@K and precision@K
def RecallPrecision_ATk(
    groundTruth: List[List[int]], r: Tensor, k: int
) -> Tuple[float, float]:
    """Computers recall @ k and precision @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (intg): determines the top k items to compute precision and recall on

    Returns:
        tuple: recall @ k, precision @ k
    """
    num_correct_pred = t.sum(r, dim=-1)  # number of correctly predicted items per user
    # number of items liked by each user in the test set
    user_num_liked = t.Tensor([len(groundTruth[i]) for i in range(len(groundTruth))])
    recall = t.mean(num_correct_pred / user_num_liked)
    precision = t.mean(num_correct_pred) / k
    return recall.item(), precision.item()


# computes NDCG@K
def NDCGatK_r(groundTruth: List[List[int]], r: Tensor, k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (int): determines the top k items to compute ndcg on

    Returns:
        float: ndcg @ k
    """
    assert len(r) == len(groundTruth)

    test_matrix = t.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = t.sum(max_r * 1.0 / t.log2(t.arange(2, k + 2)), axis=1)
    dcg = r * (1.0 / t.log2(t.arange(2, k + 2)))
    dcg = t.sum(dcg, axis=1)
    idcg[idcg == 0.0] = 1.0
    ndcg = dcg / idcg
    ndcg[t.isnan(ndcg)] = 0.0
    return t.mean(ndcg).item()


# wrapper function to get evaluation metrics
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
    user_embedding = model.users_emb.weight.to("cpu")
    item_embedding = model.items_emb.weight.to("cpu")

    excluded_edges_per_user = create_edges_dict_indexed_by_user(
        t.cat(exclude_edge_indices, dim=1)
    )

    # get all unique users in evaluated split
    users = edge_index[0].unique()

    # get the top k recommended items for each user
    top_K_items = {}
    for user in users:
        top_K_items[user.item()] = get_topk_items(
            user_embedding, item_embedding, user.item(), k, excluded_edges_per_user
        )

    test_user_pos_items = create_edges_dict_indexed_by_user(edge_index)

    # convert test user pos items dictionary into a list
    test_user_pos_items_list = [test_user_pos_items[user.item()] for user in users]

    # determine the correctness of topk predictions
    r = []
    for user in users:
        user = user.item()
        ground_truth_items = test_user_pos_items[user]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = t.Tensor(np.array(r).astype("float"))

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)

    return recall, precision, ndcg


def get_topk_items(
    user_embeddings: Tensor,
    item_embeddings: Tensor,
    user_id: Tensor,
    k: int,
    exclude_edge_indices: dict,
) -> Tensor:
    rating = t.matmul(user_embeddings[user_id], item_embeddings.T)
    _, topK = t.topk(rating, k=k)
    return topK
