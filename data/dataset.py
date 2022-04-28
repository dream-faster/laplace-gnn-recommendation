import torch
import math
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch import Tensor
from typing import Union

#


class GraphDataset(InMemoryDataset):
    def __init__(self, edge_dir, graph_dir):
        self.edges = torch.load(edge_dir)
        self.graph = torch.load(graph_dir)

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, idx: int) -> Union[Data, HeteroData]:
        all_edges = torch.tensor(self.edges[idx])

        """ Prepare features and edges """
        user_features = self.graph["customer"].x[idx]
        article_features = torch.empty(
            size=(
                len(all_edges),
                self.graph["article"].x[self.edges[0][0]].shape[0],
            )
        )
        for i in range(len(all_edges)):
            article_features[i] = self.graph["article"].x[i]

        samp_cut = max(
            1, math.floor(len(all_edges) / 3)
        )  # This could be problematic if num_edges for a user is less than 2

        positive_edges_flat = all_edges[:samp_cut]
        negative_edges_flat = self.__get_negative_edges(
            positive_edges_flat, idx, num_negative_edges=len(positive_edges_flat)
        )

        # Add customer id to the tensor
        positive_edges = torch.stack(
            [torch.tensor([idx]).repeat(len(positive_edges_flat)), positive_edges_flat],
            dim=0,
        )
        negative_edges = torch.stack(
            [torch.tensor([idx]).repeat(len(negative_edges_flat)), negative_edges_flat]
        )
        pos_neg_edges = torch.cat([positive_edges, negative_edges], dim=1)

        # Create labels that identify negative from positive edges
        labels = torch.stack(
            [torch.ones(positive_edges.shape[1]), torch.zeros(negative_edges.shape[1])],
            dim=1,
        )

        """ Create Data """
        data = HeteroData()
        edge_key = ("customer", "buys", "article")
        rev_edge_key = ("article", "rev_buys", "customer")
        data["customer"].x = torch.unsqueeze(user_features, dim=0)
        data["article"].x = article_features

        # Add original directional edges
        data[edge_key].edge_index = all_edges
        data[edge_key].edge_label_index = pos_neg_edges
        data[edge_key].edge_label = labels

        # Add reverse edges
        reverse_key = torch.LongTensor([1, 0])
        data[rev_edge_key].edge_index = all_edges[reverse_key]
        data[rev_edge_key].edge_label_index = pos_neg_edges[reverse_key]
        data[rev_edge_key].edge_label = labels
        return data

    def __get_negative_edges(
        self, positive_edges: Tensor, idx: int, num_negative_edges: int = 10
    ) -> Tensor:
        all_ids = self.graph[("customer", "buys", "article")].edge_index
        # Get the biggest value available in articles
        id_max = torch.max(all_ids, dim=1)[0][1]

        # Create list of potential negative edges, filter out positive edges
        combined = torch.cat((torch.range(start=0, end=id_max), positive_edges))
        uniques, counts = combined.unique(return_counts=True)
        difference = uniques[counts == 1]

        # Randomly sample negative edges
        negative_edges = difference[torch.randperm(difference.nelement())][
            :num_negative_edges
        ]

        return negative_edges
