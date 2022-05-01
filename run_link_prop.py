# LinkProp and LinkProp-Multi from the paper:
# Revisiting Neighborhood-based Link Prediction for Collaborative Filtering
# https://arxiv.org/abs/2203.15789

import numpy as np
import torch
from torch_sparse import SparseTensor, matmul
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import ndcg_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator


def load_data():
    # try:
    #     return torch.load("data/derived/link_prop.pt")
    # except FileNotFoundError:
    #     pass

    customers = pd.read_csv("data/original/customers.csv")
    articles = pd.read_csv("data/original/articles.csv")
    transactions = pd.read_csv("data/original/transactions_train.csv")
    customers.reset_index()
    articles.reset_index()

    # get rid of customers that have no transactions
    # customers = customers.merge(
    #     pd.DataFrame(transactions["customer_id"].unique(), columns=["customer_id"]),
    #     on="customer_id",
    #     how="inner",
    # )
    # # get rid of articles that have no transactions
    # articles = articles.merge(
    #     pd.DataFrame(transactions["article_id"].unique(), columns=["article_id"]),
    #     on="article_id",
    #     how="inner",
    # )

    src = list(customers["customer_id"])
    src_map = {v: k for k, v in enumerate(src)}
    dest = list(articles["article_id"])
    dest_map = {v: k for k, v in enumerate(dest)}

    transactions.reset_index()
    transactions["src"] = transactions["customer_id"].map(src_map)
    transactions["dest"] = transactions["article_id"].map(dest_map)

    # shuffle transactions
    edge_index = torch.tensor(
        transactions[["src", "dest"]].sample(frac=1).values, dtype=torch.long
    ).T

    # create user-item interaction matrix
    M = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=torch.ones(len(transactions), dtype=torch.float),
        sparse_sizes=(len(src), len(dest)),
    ).coalesce("max")

    torch.save((M, src, dest, src_map, dest_map), "data/derived/link_prop.pt")

    return M, src, dest, src_map, dest_map


def split_data(start, count, M):
    """Selects a range of users"""
    return M[start : start + count, :].coalesce("max").to_dense()


def sample_edges(target, ratio=0.5):
    """Creates data by dropping random edges.
    Guarantees remaining users have at least one edge"""
    # mask nodes with degree <= 1 (cloning since you cannot mask twice at the same time)
    user_deg, item_deg = target.sum(dim=0), target.sum(dim=1)
    edges = target.clone()
    edges[:, user_deg <= 1] = 0
    edges[item_deg <= 1, :] = 0
    # get edge indices
    row, col = edges.nonzero(as_tuple=True)
    assert (target[row, col] == 1).all(), "row, col should be all 1"

    # sample some % of edges
    sample_mask = torch.randint(0, len(row), (int(len(row) * ratio),))
    # clear max 1 edge for each user and item
    data = target.clone()
    cleared = {}
    for i, j in torch.stack((row[sample_mask], col[sample_mask])).T:
        if i.item() not in cleared and j.item() not in cleared:
            assert data[i, j] == 1, "a,b should be 1"
            cleared[i.item()] = True
            cleared[j.item()] = True
            data[i, j] = 0.0

    # assert (data.sum(dim=1) >= 1).all(), "should have at least one item"
    # assert (data.sum(dim=0) >= 1).all(), "should have at least one user"
    return data, target


def mean_average_precision(y_true, y_pred, k=12):
    """Courtesy of https://www.kaggle.com/code/george86/calculate-map-12-fast-faster-fastest"""
    # compute the Rel@K for all items
    rel_at_k = np.zeros((len(y_true), k), dtype=int)

    # collect the intersection indexes (for the ranking vector) for all pairs
    for idx, (truth, pred) in enumerate(zip(y_true, y_pred)):
        _, _, inter_idxs = np.intersect1d(
            truth, pred[:k], assume_unique=True, return_indices=True
        )
        rel_at_k[idx, inter_idxs] = 1

    # Calculate the intersection counts for all pairs
    intersection_count_at_k = rel_at_k.cumsum(axis=1)

    # we have the same denominator for all ranking vectors
    ranks = np.arange(1, k + 1, 1)

    # Calculating the Precision@K for all Ks for all pairs
    precisions_at_k = intersection_count_at_k / ranks
    # Multiply with the Rel@K for all pairs
    precisions_at_k = precisions_at_k * rel_at_k

    # Calculate the average precisions @ K for all pairs
    average_precisions_at_k = precisions_at_k.mean(axis=1)

    # calculate the final MAP@K
    map_at_k = average_precisions_at_k.mean()

    return map_at_k


class LinkPropMulti(BaseEstimator):
    def __init__(self, rounds, k, alpha, beta, gamma, delta):
        super().__init__()
        self.rounds, self.k, self.alpha, self.beta, self.gamma, self.delta = (
            rounds,
            k,
            alpha,
            beta,
            gamma,
            delta,
        )
        self.user_degrees = None
        self.item_degrees = None
        self.L = None
        self.M = None
        self.M_target = None

    def fit(self, X, y):
        return self

    def fit_for_score(self, M, target):
        # reset model
        self.L = None

        # get node degrees
        self.user_degrees = M.sum(dim=1)
        self.item_degrees = M.sum(dim=0)

        # exponentiate degrees by model params
        user_alpha = self.user_degrees ** (-self.alpha)
        item_beta = self.item_degrees ** (-self.beta)
        user_gamma = self.user_degrees ** (-self.gamma)
        item_delta = self.item_degrees ** (-self.delta)

        # get rid of inf from 1/0
        user_alpha[torch.isinf(user_alpha)] = 0.0
        item_beta[torch.isinf(item_beta)] = 0.0
        user_gamma[torch.isinf(user_gamma)] = 0.0
        item_delta[torch.isinf(item_delta)] = 0.0

        # outer products
        alpha_beta = user_alpha.reshape((-1, 1)) * item_beta
        gamma_delta = user_gamma.reshape((-1, 1)) * item_delta

        # hadamard products
        M_alpha_beta = M * alpha_beta
        M_gamma_delta = M * gamma_delta
        self.L = M_alpha_beta.matmul(M.T).matmul(M_gamma_delta)

        # get top k new links
        target_pred = self.predict_k(M, self.k)

        # # with the new links recalculate and store node degrees for next round
        # M_new = (M.clone() + target_pred).clamp(max=1)
        # self.user_degrees = M_new.sum(dim=1)
        # self.item_degrees = M_new.sum(dim=0)

        return M, target, target_pred

    def predict_k(self, M, k):
        """Return top k new links
        M: should be the same data as in fit
        k: number of new links to return
        """
        # take observered links out of predictions
        target_pred = (self.L - (M == 1).float() * 100000).clamp(min=0)

        # select top k links for each user
        for i, col in enumerate(target_pred.topk(k, dim=1)[1]):
            target_pred[i, col] = 1

        # clear predictions, keep only new links
        target_pred = (target_pred == 1).float()

        return target_pred

    def predict(self, users_ix, k):
        # return self.L[users_ix, :].topk(k, dim=1)[1]
        pass

    def score(self, X, y):
        # process data
        M, target, target_pred = self.fit_for_score(X, y)
        # take observed out of ground truth
        target_true = (target - (M == 1).float() * 100000).clamp(min=0)

        return ndcg_score(target_true, target_pred)

    def score_mapk(self, X, y):
        M, target, target_pred = self.fit_for_score(X, y)
        # take observed out of ground truth
        # TODO this creates users with 0 items
        # TODO also sampling only makes one new item per user
        target_true = (target - (M == 1).float() * 100000).clamp(min=0)
        # only calculating for users with at least one item in true target
        target_t = target_true.topk(self.k, dim=1)[1][
            (target_true.topk(self.k, dim=1)[0] > 0).any(dim=1)
        ]
        target_p = target_pred.topk(self.k, dim=1)[1][
            (target_true.topk(self.k, dim=1)[0] > 0).any(dim=1)
        ]
        return (
            mean_average_precision(target_t, target_p, k=self.k),
            target_pred.topk(self.k, dim=1)[1],
        )


M, src, dest, src_map, dest_map = load_data()
# split data
train_size, val_size, test_size = 2000, 1000, 1000

# TODO should we use the original node degrees?
scores = []
preds = torch.tensor([])
total = len(src) // train_size
for i in tqdm(range(total)):
    batch_size = train_size if i < total - 1 else len(src) - train_size * i
    data, target = sample_edges(split_data(i * train_size, batch_size, M), 0.4)
    linkProp = LinkPropMulti(rounds=1, k=12, alpha=0, beta=0, gamma=0, delta=0.5)
    score, pred = linkProp.score_mapk(data, target)
    scores.append(score)
    preds = torch.cat((preds, pred))
    print(np.array(scores).mean())
submission = pd.DataFrame(preds.apply_(lambda x: dest[int(x)])).astype("string")
submission["prediction"] = submission[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]].agg(
    " ".join, axis=1
)
submission["customer_id"] = src[: len(submission)]
submission[["customer_id", "prediction"]].to_csv(
    "data/derived/submission.csv", index=False
)
# param_grid = [
#     {
#         "alpha": [0.0, 0.5],
#         "beta": [0.0, 0.5],
#         "gamma": [0.0, 0.5],
#         "delta": [0.0, 0.5],
#         "k": [3],
#         "rounds": [1],
#     },
# ]
# linkProp = LinkPropMulti(rounds=1, k=12, alpha=0, beta=0, gamma=0, delta=0)
# clf = GridSearchCV(
#     linkProp,
#     param_grid=param_grid,
#     verbose=2,
#     return_train_score=True,
# )
# clf.fit(data, target)
# sorted(clf.cv_results_.keys())
# print(clf.cv_results_)
