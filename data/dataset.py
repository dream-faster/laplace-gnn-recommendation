import torch
import math
from torch_geometric.data import HeteroData, InMemoryDataset

#


class GraphDataset(InMemoryDataset):
    def __init__(self, edge_dir, graph_dir):
        self.edges = torch.load(edge_dir)
        self.graph = torch.load(graph_dir)

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, idx: int):
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

        cut = max(
            1, math.floor(len(positive_edges) / 3)
        )  # This could be problematic if num_edges for a user is less than 2

        # We need to add negative edges aswell.
        data = HeteroData()
        data["customer"].x = torch.unsqueeze(user_features, dim=0)
        data["article"].x = article_features
        data["customer", "buys", "article"].edge_index = positive_edges
        data["customer", "buys", "article"].edge_label_index = positive_edges
        data["customer", "buys", "article"].edge_label = torch.stack(
            [torch.ones(cut), torch.zeros(len(positive_edges) - cut)], dim=1
        )

        data["customer", "rev_buys", "article"].edge_index = reversed_edges
        data["customer", "rev_buys", "article"].edge_label_index = reversed_edges
        data["customer", "buys", "article"].edge_label = torch.stack(
            [torch.ones(cut), torch.zeros(len(positive_edges) - cut)], dim=1
        )
        return data
