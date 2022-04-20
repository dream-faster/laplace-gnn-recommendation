# import required modules
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim, Tensor

from torch_sparse import SparseTensor, matmul

from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import download_url, extract_zip


from torch_geometric.typing import Adj
from model.lightgcn import LightGCN
import json

"""# Loading the Dataset

We load the dataset and set ratings >=4 on a 0.5 ~ 5 scale as an edge between users and movies.

We split the edges of the graph using a 80/10/10 train/validation/test split.
"""


def split(edge_index):
    # edge_index = load_edge_csv(
    #     rating_path,
    #     src_index_col="userId",
    #     src_mapping=user_mapping,
    #     dst_index_col="movieId",
    #     dst_mapping=movie_mapping,
    #     link_index_col="rating",
    #     rating_threshold=4,
    # )
    # split the edges of the graph using a 80/10/10 train/validation/test split

    num_interactions = edge_index.shape[1]
    all_indices = [i for i in range(num_interactions)]

    train_indices, test_indices = train_test_split(
        all_indices, test_size=0.2, random_state=1
    )
    val_indices, test_indices = train_test_split(
        test_indices, test_size=0.5, random_state=1
    )

    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]

    return train_edge_index, val_edge_index, test_edge_index, edge_index


def read_json(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)


def both_indexes_from_zero(edge_index):
    new_edge_index = torch.clone(edge_index)
    new_edge_index[1] = new_edge_index[1] - (torch.max(new_edge_index[0]) + 1)

    return new_edge_index


def create_dataloaders_lightgcn():
    data = torch.load("data/derived/graph.pt").to_homogeneous()

    customer_id_map = read_json("data/derived/customer_id_map_forward.json")
    article_id_map = read_json("data/derived/article_id_map_forward.json")

    # user_mapping = load_node_csv(rating_path, index_col="userId")
    # movie_mapping = load_node_csv(movie_path, index_col="movieId")
    # num_users, num_movies = len(user_mapping), len(movie_mapping)

    num_users, num_movies = len(customer_id_map), len(article_id_map)

    edge_index = both_indexes_from_zero(data.edge_index)
    train_edge_index, val_edge_index, test_edge_index, edge_index = split(edge_index)

    # convert edge indices into Sparse Tensors: https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html
    train_sparse_edge_index = SparseTensor(
        row=train_edge_index[0],
        col=train_edge_index[1],
        sparse_sizes=(num_users + num_movies, num_users + num_movies),
    )
    val_sparse_edge_index = SparseTensor(
        row=val_edge_index[0],
        col=val_edge_index[1],
        sparse_sizes=(num_users + num_movies, num_users + num_movies),
    )
    test_sparse_edge_index = SparseTensor(
        row=test_edge_index[0],
        col=test_edge_index[1],
        sparse_sizes=(num_users + num_movies, num_users + num_movies),
    )

    return (
        train_sparse_edge_index,
        val_sparse_edge_index,
        test_sparse_edge_index,
        train_edge_index,
        val_edge_index,
        test_edge_index,
        edge_index,
        customer_id_map,
        article_id_map,
        num_users,
        num_movies,
    )


# function which random samples a mini-batch of positive and negative samples
def sample_mini_batch(batch_size, edge_index):
    """Randomly samples indices of a minibatch given an adjacency matrix

    Args:
        batch_size (int): minibatch size
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        tuple: user indices, positive item indices, negative item indices
    """
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    indices = random.choices([i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices
