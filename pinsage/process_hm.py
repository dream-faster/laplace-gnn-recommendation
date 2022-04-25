"""
Script that reads from raw MovieLens-1M data and dumps into a pickle
file the following:

* A heterogeneous graph with categorical features.
* A list with all the movie titles.  The movie titles correspond to
  the movie nodes in the heterogeneous graph.

This script exemplifies how to prepare tabular data with textual
features.  Since DGL graphs do not store variable-length features, we
instead put variable-length features into a more suitable container
(e.g. torchtext to handle list of texts)
"""

import os
import re
import argparse
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as ssp
import dgl
import torch
import torchtext
from pinsage.builder import PandasGraphBuilder
from pinsage.data_utils import *


def process_hm():

    output_path = "dataset.pkl"

    print("| Loading Graph...")
    g = torch.load("graph.pt")

    print("| Loading Transactions...")
    transactions = pd.read_parquet("transactions_train.parquet")[
        : int(g.num_edges() / 2)
    ]

    g.edges["buys"].data["timestamp"] = torch.LongTensor(
        pd.DatetimeIndex(transactions["t_dat"]).asi8
    )
    g.edges["rev_buys"].data["timestamp"] = torch.LongTensor(
        pd.DatetimeIndex(transactions["t_dat"]).asi8
    )
    # Train-validation-test split
    # This is a little bit tricky as we want to select the last interaction for test, and the
    # second-to-last interaction for validation.
    print("| Creating Train-Test Split...")
    train_indices, val_indices, test_indices = train_test_split_by_time(
        transactions, "t_dat", "customer_id"
    )

    print("| Building Train Graph...")
    # Build the graph with training interactions only.
    train_g = build_train_graph(
        g, train_indices, "customer", "article", "buys", "rev_buys"
    )
    assert train_g.out_degrees(etype="buys").min() > 0

    # Build the user-item sparse matrix for validation and test set.
    val_matrix, test_matrix = build_val_test_matrix(
        g, val_indices, test_indices, "customer", "article", "buys"
    )

    ## Dump the graph and the datasets
    print("| Creating dataset...")
    dataset = {
        "train-graph": train_g,
        "val-matrix": val_matrix,
        "test-matrix": test_matrix,
        "item-texts": {},
        "item-images": None,
        "user-type": "customer",
        "item-type": "article",
        "user-to-item-type": "buys",
        "item-to-user-type": "rev_buys",
        "timestamp-edge-column": "timestamp",
    }

    print("| Saving dataset...")
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    process_hm()
