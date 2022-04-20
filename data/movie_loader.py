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

"""# Loading the Dataset

We load the dataset and set ratings >=4 on a 0.5 ~ 5 scale as an edge between users and movies.

We split the edges of the graph using a 80/10/10 train/validation/test split.
"""

# download the dataset
url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
extract_zip(download_url(url, "./data/movie/raw"), "./data/movie")

# load user and movie nodes
def load_node_csv(path, index_col):
    """Loads csv containing node information

    Args:
        path (str): path to csv file
        index_col (str): column name of index column

    Returns:
        dict: mapping of csv row to node id
    """
    df = pd.read_csv(path, index_col=index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    return mapping


# load edges between users and movies
def load_edge_csv(
    path,
    src_index_col,
    src_mapping,
    dst_index_col,
    dst_mapping,
    link_index_col,
    rating_threshold=4,
):
    """Loads csv containing edges between users and items

    Args:
        path (str): path to csv file
        src_index_col (str): column name of users
        src_mapping (dict): mapping between row number and user id
        dst_index_col (str): column name of items
        dst_mapping (dict): mapping between row number and item id
        link_index_col (str): column name of user item interaction
        rating_threshold (int, optional): Threshold to determine positivity of edge. Defaults to 4.

    Returns:
        torch.Tensor: 2 by N matrix containing the node ids of N user-item edges
    """
    df = pd.read_csv(path)
    edge_index = None
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_attr = (
        torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long)
        >= rating_threshold
    )

    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])

    return torch.tensor(edge_index)


def split(user_mapping, movie_mapping, rating_path):
    edge_index = load_edge_csv(
        rating_path,
        src_index_col="userId",
        src_mapping=user_mapping,
        dst_index_col="movieId",
        dst_mapping=movie_mapping,
        link_index_col="rating",
        rating_threshold=4,
    )
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


def create_dataloaders(movie_path, rating_path):
    user_mapping = load_node_csv(rating_path, index_col="userId")
    movie_mapping = load_node_csv(movie_path, index_col="movieId")
    num_users, num_movies = len(user_mapping), len(movie_mapping)
    train_edge_index, val_edge_index, test_edge_index, edge_index = split(
        user_mapping, movie_mapping, rating_path
    )

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
        user_mapping,
        movie_mapping,
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
