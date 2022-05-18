import torch as t
from torch import Tensor
from typing import List, Tuple


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
    num_correct_pred = t.sum(
        r, dim=-1
    ).float()  # number of correctly predicted items per user
    # number of items liked by each user in the test set
    user_num_liked = t.Tensor([len(row) for row in groundTruth])
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
