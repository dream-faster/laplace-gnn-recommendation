import pickle
import pandas as pd
import dgl
import torch
from pinsage.data_utils import *
from utils.constants import Constants


def process_hm():

    output_path = "data/derived/pinsage_dataset.pkl"

    print("| Loading Graph...")
    g = torch.load("data/derived/test_graph.pt")

    print("| Loading Transactions...")
    transactions = pd.read_parquet("data/original/transactions_splitted.parquet")

    g.edges["buys"].data["timestamp"] = torch.LongTensor(
        pd.DatetimeIndex(transactions["t_dat"]).asi8
    )
    g.edges["rev_buys"].data["timestamp"] = torch.LongTensor(
        pd.DatetimeIndex(transactions["t_dat"]).asi8
    )

    print("| Creating Train-Test Split...")
    train_indices = transactions["train_mask"].to_numpy().nonzero()[0]
    val_indices = transactions["val_mask"].to_numpy().nonzero()[0]
    test_indices = transactions["test_mask"].to_numpy().nonzero()[0]

    print("| Building Train Graph...")
    train_g = torch.load("data/derived/train_graph.pt")
    assert train_g.out_degrees(etype="buys").min() > 0

    # Build the user-item sparse matrix for validation and test set.
    val_matrix, test_matrix = build_val_test_matrix(
        g, val_indices, test_indices, Constants.node_user, Constants.node_item, "buys"
    )

    ## Dump the graph and the datasets
    print("| Creating dataset...")
    dataset = {
        "train-graph": train_g,
        "val-matrix": val_matrix,
        "test-matrix": test_matrix,
        "item-texts": {},
        "item-images": None,
        "user-type": Constants.node_user,
        "item-type": Constants.node_item,
        "user-to-item-type": "buys",
        "item-to-user-type": "rev_buys",
        "timestamp-edge-column": "timestamp",
    }

    print("| Saving dataset...")
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":

    process_hm()
