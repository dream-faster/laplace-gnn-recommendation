import pandas as pd
from torch.utils.data import Dataset
import torch


class GraphDataset(Dataset):
    def __init__(self, edge_dir, graph_dir, transform=None, target_transform=None):
        self.edges = torch.load(edge_dir)
        self.graph = torch.load(graph_dir)

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        connected_articles = self.edges[idx]
        user_features = self.graph["customer"].x[idx]

        for edge in connected_articles:
            article_features = self.graph["article"].x[edge]

        return connected_articles, user_features, article_features
