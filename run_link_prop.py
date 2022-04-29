# LinkProp and LinkProp-Multi from the paper:
# Revisiting Neighborhood-based Link Prediction for Collaborative Filtering
# https://arxiv.org/abs/2203.15789

import torch
from torch_sparse import SparseTensor, matmul
import pandas as pd
import tqdm
from sklearn.metrics import ndcg_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator

device = "cpu"
train_size, val_size, test_size = 1000, 1000, 1000

customers = pd.read_csv("data/original/customers.csv")
articles = pd.read_csv("data/original/articles.csv")
transactions = pd.read_csv("data/original/transactions_train.csv")
customers.reset_index()
articles.reset_index()

# get rid of customers that have no transactions
customers = customers.merge(
    pd.DataFrame(transactions["customer_id"].unique(), columns=["customer_id"]),
    on="customer_id",
    how="inner",
)

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


def split_data(start, count, M):
    """Selects a range of users and filters out newly unconnected items"""
    item_deg = M[
        start : start + count,
    ].sum(dim=0)
    return M[start : start + count, item_deg > 0].coalesce("max").to_dense()


def negative_sample(y, ratio=0.5):
    """Creates negative sample by dropping random edges.
    Guarantees remaining items and users have at least one edge"""
    # mask nodes with degree <= 1 (cloning since you cannot mask twice at the same time)
    user_deg, item_deg = y.sum(dim=0), y.sum(dim=1)
    edges = y.clone()
    edges[:, user_deg <= 1] = 0
    edges[item_deg <= 1, :] = 0
    # get edge indices
    row, col = edges.nonzero(as_tuple=True)
    assert (y[row, col] == 1).all(), "row, col should be all 1"

    # sample some % of edges
    sample_mask = torch.randint(0, len(row), (int(len(row) * ratio),))
    # clear max 1 edge for each user and item
    x = y.clone()
    cleared = {}
    for i, j in torch.stack((row[sample_mask], col[sample_mask])).T:
        if i.item() not in cleared and j.item() not in cleared:
            assert x[i, j] == 1, "a,b should be 1"
            cleared[i.item()] = True
            cleared[j.item()] = True
            x[i, j] = 0.0

    assert (x.sum(dim=1) >= 1).all(), "should have at least one item"
    assert (x.sum(dim=0) >= 1).all(), "should have at least one user"
    return x, y


# split data
# TODO should we use the original node degrees?
train, test, val = (
    split_data(0, train_size, M),
    split_data(train_size, val_size, M),
    split_data(train_size + val_size, test_size, M),
)

# negative samples
x_train, y_train = negative_sample(train, 0.4)
x_val, y_val = negative_sample(val, 0.4)
x_test, y_test = negative_sample(test, 0.4)


class LinkPropMulti:
    def __init__(self, M, M_true, rounds=1):
        self.M = M
        self.M_true = M_true
        self.rounds = rounds
        self.L = M
        self.user_degrees = M.sum(dim=1)
        self.item_degrees = M.sum(dim=0)

    def __call__(self, alpha, beta, gamma, delta):
        # exponentiate degrees by model params
        user_alpha = self.user_degrees ** (-alpha)
        item_beta = self.item_degrees ** (-beta)
        user_gamma = self.user_degrees ** (-gamma)
        item_delta = self.item_degrees ** (-delta)

        # get rid of inf from 1/0
        user_alpha[torch.isinf(user_alpha)] = 0.0
        item_beta[torch.isinf(item_beta)] = 0.0
        user_gamma[torch.isinf(user_gamma)] = 0.0
        item_delta[torch.isinf(item_delta)] = 0.0

        # outer products
        alpha_beta = user_alpha.reshape((-1, 1)) * item_beta
        gamma_delta = user_gamma.reshape((-1, 1)) * item_delta

        # hadamard products
        M_alpha_beta = self.M * alpha_beta
        M_gamma_delta = self.M * gamma_delta
        self.L = M_alpha_beta.matmul(self.M.T).matmul(M_gamma_delta)

        # return the link propagated interaction matrix
        return self.L

    def predict(self, users_ix, k):
        return self.L[users_ix, :].topk(k, dim=1)[1]

    def fit(self, alpha, beta, gamma, delta, k=3, t=0.05, update_M=False):
        L = self(alpha, beta, gamma, delta)

        # take observered links out of predictions
        y_pred = (L - (self.M == 1).float() * 100000).clamp(min=0)

        # number_of_links = self.item_degrees.sum()
        # top_link_ix = L.topk(int(number_of_links * t))[1]

        # select top k links for each user
        for i, col in enumerate(y_pred.topk(k, dim=1)[1]):
            y_pred[i, col] = 1

        # clear predictions, keep only new links
        y_pred = (y_pred == 1).float()

        # take observed out of ground truth
        y_true = (self.M_true - (self.M == 1).float() * 100000).clamp(min=0)

        # recalculate and store node degrees for next round
        # based on new top links
        M_new = self.M if update_M else self.M.clone()
        M_new = (M_new + y_pred).clamp(max=1)
        self.user_degrees = M_new.sum(dim=1)
        self.item_degrees = M_new.sum(dim=0)

        # calculate loss on new links
        return ndcg_score(y_true, y_pred)


parameters = {"alpha": 0.0, "beta": 0.67, "gamma": 0.5, "delta": 0.5}
linkProp = LinkPropMulti(x_train, y_train)
print(linkProp.fit(**parameters))
# print("orig:", ndcg_score(y_train, linkProp.M))
# print("pred:", ndcg_score(y_train, linkProp.L))
# svc = svm.SVC()
# clf = GridSearchCV(LinkProp(user_degrees, item_degrees, M), parameters)
# clf.fit(iris.data, iris.target)
# GridSearchCV(estimator=SVC(),
#              param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})
# sorted(clf.cv_results_.keys())
