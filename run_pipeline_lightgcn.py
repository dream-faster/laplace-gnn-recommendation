# Commented out IPython magic to ensure Python compatibility.
# # Install required packages.
# %%capture
# !pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
# !pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
# !pip install torch-geometric
# !pip install -q git+https://github.com/snap-stanford/deepsnap.git
# !pip install -U -q PyDrive

"""# Implementing a Recommender System using LightGCN

In this colab, we explain how to set up a graph recommender system using the [LighGCN](https://arxiv.org/abs/2002.02126) model. Specifically, we apply LightGCN to a movie recommendation task using [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/).

We use the [MovieLens](https://grouplens.org/datasets/movielens/) (*small*) dataset which has 100,000 ratings applied to 9,000 movies by 600 users. 

Our implementation was inspired by the following documentation and repositories:
- https://github.com/gusye1234/LightGCN-PyTorch
- https://www.kaggle.com/dipanjandas96/lightgcn-pytorch-from-scratch
- https://pytorch-geometric.readthedocs.io/en/latest/notes/load_csv.html
"""

# import required modules

import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import optim

from torch_geometric.utils import structured_negative_sampling


from torch_geometric.typing import Adj
from model.lightgcn import LightGCN
from data.movie_loader import create_dataloaders, sample_mini_batch
from data.lightgcn_loader import create_dataloaders_lightgcn, sample_mini_batch
from utils.lightgcn_metrics import (
    get_metrics,
    bpr_loss,
    RecallPrecision_ATk,
    NDCGatK_r,
    get_user_positive_items,
)

movie_path = "./data/movie/ml-latest-small/movies.csv"
rating_path = "./data/movie/ml-latest-small/ratings.csv"


# wrapper function to evaluate model
def evaluation(
    model, edge_index, sparse_edge_index, exclude_edge_indices, k, lambda_val
):
    """Evaluates model loss and metrics including recall, precision, ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        sparse_edge_index (sparseTensor): sparse adjacency matrix for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on
        lambda_val (float): determines lambda for bpr loss

    Returns:
        tuple: bpr loss, recall @ k, precision @ k, ndcg @ k
    """
    # get embeddings
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
        sparse_edge_index
    )
    edges = structured_negative_sampling(edge_index, contains_neg_self_loops=False)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    users_emb_final, users_emb_0 = (
        users_emb_final[user_indices],
        users_emb_0[user_indices],
    )
    pos_items_emb_final, pos_items_emb_0 = (
        items_emb_final[pos_item_indices],
        items_emb_0[pos_item_indices],
    )
    neg_items_emb_final, neg_items_emb_0 = (
        items_emb_final[neg_item_indices],
        items_emb_0[neg_item_indices],
    )

    loss = bpr_loss(
        users_emb_final,
        users_emb_0,
        pos_items_emb_final,
        pos_items_emb_0,
        neg_items_emb_final,
        neg_items_emb_0,
        lambda_val,
    ).item()

    recall, precision, ndcg = get_metrics(model, edge_index, exclude_edge_indices, k)

    return loss, recall, precision, ndcg


def train():
    """# Training

    Your test set performance should be in line with the following (*K=20*):

    *Recall@K: 0.13, Precision@K: 0.045, NDCG@K: 0.10*
    """
    (
        train_sparse_edge_index,
        val_sparse_edge_index,
        test_sparse_edge_index,
        train_edge_index,
        val_edge_index,
        test_edge_index,
        edge_index,
        user_mapping,
        movie_mapping,
        num_users,
        num_movies,
    ) = create_dataloaders_lightgcn()
    # (
    #     train_sparse_edge_index,
    #     val_sparse_edge_index,
    #     test_sparse_edge_index,
    #     train_edge_index,
    #     val_edge_index,
    #     test_edge_index,
    #     edge_index,
    #     user_mapping,
    #     movie_mapping,
    #     num_users,
    #     num_movies,
    # ) = create_dataloaders(movie_path, rating_path)
    # define contants
    ITERATIONS = 10000
    BATCH_SIZE = 1024
    LR = 1e-3
    ITERS_PER_EVAL = 200
    ITERS_PER_LR_DECAY = 200
    K = 20
    LAMBDA = 1e-6

    # setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")

    model = LightGCN(num_users, num_movies)
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    edge_index = edge_index.to(device)
    train_edge_index = train_edge_index.to(device)
    train_sparse_edge_index = train_sparse_edge_index.to(device)

    val_edge_index = val_edge_index.to(device)
    val_sparse_edge_index = val_sparse_edge_index.to(device)

    # training loop
    train_losses = []
    val_losses = []

    for iter in range(ITERATIONS):
        # forward propagation
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
            train_sparse_edge_index
        )

        # mini batching
        user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(
            BATCH_SIZE, train_edge_index
        )
        user_indices, pos_item_indices, neg_item_indices = (
            user_indices.to(device),
            pos_item_indices.to(device),
            neg_item_indices.to(device),
        )
        users_emb_final, users_emb_0 = (
            users_emb_final[user_indices],
            users_emb_0[user_indices],
        )
        pos_items_emb_final, pos_items_emb_0 = (
            items_emb_final[pos_item_indices],
            items_emb_0[pos_item_indices],
        )
        neg_items_emb_final, neg_items_emb_0 = (
            items_emb_final[neg_item_indices],
            items_emb_0[neg_item_indices],
        )

        # loss computation
        train_loss = bpr_loss(
            users_emb_final,
            users_emb_0,
            pos_items_emb_final,
            pos_items_emb_0,
            neg_items_emb_final,
            neg_items_emb_0,
            LAMBDA,
        )

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if iter % ITERS_PER_EVAL == 0:
            model.eval()
            val_loss, recall, precision, ndcg = evaluation(
                model,
                val_edge_index,
                val_sparse_edge_index,
                [train_edge_index],
                K,
                LAMBDA,
            )
            print(
                f"[Iteration {iter}/{ITERATIONS}] train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss, 5)}, val_recall@{K}: {round(recall, 5)}, val_precision@{K}: {round(precision, 5)}, val_ndcg@{K}: {round(ndcg, 5)}"
            )
            train_losses.append(train_loss.item())
            val_losses.append(val_loss)
            model.train()

        if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
            scheduler.step()

    iters = [iter * ITERS_PER_EVAL for iter in range(len(train_losses))]
    plt.plot(iters, train_losses, label="train")
    plt.plot(iters, val_losses, label="validation")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("training and validation loss curves")
    plt.legend()
    plt.show()

    # evaluate on test set
    model.eval()
    test_edge_index = test_edge_index.to(device)
    test_sparse_edge_index = test_sparse_edge_index.to(device)

    test_loss, test_recall, test_precision, test_ndcg = evaluation(
        model,
        test_edge_index,
        test_sparse_edge_index,
        [train_edge_index, val_edge_index],
        K,
        LAMBDA,
    )

    print(
        f"[test_loss: {round(test_loss, 5)}, test_recall@{K}: {round(test_recall, 5)}, test_precision@{K}: {round(test_precision, 5)}, test_ndcg@{K}: {round(test_ndcg, 5)}"
    )

    """# Make New Recommendatios for a Given User"""

    model.eval()
    df = pd.read_csv(movie_path)
    movieid_title = pd.Series(df.title.values, index=df.movieId).to_dict()
    movieid_genres = pd.Series(df.genres.values, index=df.movieId).to_dict()

    user_pos_items = get_user_positive_items(edge_index)

    def make_predictions(user_id, num_recs):
        user = user_mapping[user_id]
        e_u = model.users_emb.weight[user]
        scores = model.items_emb.weight @ e_u

        values, indices = torch.topk(scores, k=len(user_pos_items[user]) + num_recs)

        movies = [
            index.cpu().item() for index in indices if index in user_pos_items[user]
        ][:num_recs]
        movie_ids = [
            list(movie_mapping.keys())[list(movie_mapping.values()).index(movie)]
            for movie in movies
        ]
        titles = [movieid_title[id] for id in movie_ids]
        genres = [movieid_genres[id] for id in movie_ids]

        print(f"Here are some movies that user {user_id} rated highly")
        for i in range(num_recs):
            print(f"title: {titles[i]}, genres: {genres[i]} ")

        print()

        movies = [
            index.cpu().item() for index in indices if index not in user_pos_items[user]
        ][:num_recs]
        movie_ids = [
            list(movie_mapping.keys())[list(movie_mapping.values()).index(movie)]
            for movie in movies
        ]
        titles = [movieid_title[id] for id in movie_ids]
        genres = [movieid_genres[id] for id in movie_ids]

        print(f"Here are some suggested movies for user {user_id}")
        for i in range(num_recs):
            print(f"title: {titles[i]}, genres: {genres[i]} ")

    USER_ID = 1
    NUM_RECS = 10

    make_predictions(USER_ID, NUM_RECS)


if __name__ == "__main__":
    train()
