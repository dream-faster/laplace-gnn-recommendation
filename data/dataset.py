import pandas as pd
from torch.utils.data import Dataset
import torch
import math


class GraphDataset(Dataset):
    def __init__(self, edge_dir, graph_dir, transform=None, target_transform=None):
        self.edges = torch.load(edge_dir)
        self.graph = torch.load(graph_dir)

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        positive_edges = torch.tensor(self.edges[idx])
        reversed_edges = positive_edges[torch.LongTensor([1, 0])]
        user_features = self.graph["customer"].x[idx]

        article_features = torch.empty(
            size=(
                len(positive_edges),
                self.graph["article"].x[self.edges[0][0]].shape[0],
            )
        )
        for i in range(len(positive_edges)):
            article_features[i] = self.graph["article"].x[i]

        x_dict = dict(customer=user_features, article=article_features)

        cut = min(
            1, math.floor(len(positive_edges) / 3)
        )  # This could be problematic if num_edges for a user is less than 2

        # We need to add negative edges aswell.

        edge_index_dict = dict()
        edge_index_dict[("customer", "buys", "article")] = positive_edges[:cut]
        edge_index_dict[("customer", "rev_buys", "article")] = reversed_edges[:cut]

        edge_label_index = dict()
        edge_label_index[("customer", "buys", "article")] = positive_edges[cut:]
        edge_label_index[("customer", "rev_buys", "article")] = reversed_edges[cut:]

        return {
            "x_dict": x_dict,
            "edge_index_dict": edge_index_dict,
            "edge_label_index": edge_label_index,
        }
