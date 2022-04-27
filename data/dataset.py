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
        connected_articles = torch.tensor(self.edges[idx])
        user_features = self.graph["customer"].x[idx]

        article_features = torch.empty(
            size=(
                len(connected_articles),
                self.graph["article"].x[self.edges[0][0]].shape[0],
            )
        )
        for i in range(len(connected_articles)):
            article_features[i] = self.graph["article"].x[i]

        return connected_articles, user_features, article_features
    