import torch
import math
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch import Tensor
from typing import Union, Optional


class GraphDataset(InMemoryDataset):
    def __init__(self, edge_dir, graph_dir):
        self.edges = torch.load(edge_dir)
        self.graph = torch.load(graph_dir)

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, idx: int) -> Union[Data, HeteroData]:
        edge_key = ("customer", "buys", "article")
        rev_edge_key = ("article", "rev_buys", "customer")

        """ Node Features """
        # Prepare user features
        user_features = self.graph["customer"].x[idx]

        # Prepare connected article features
        all_sampled_edges = torch.cat(
            [subgraph_sample_positive, sampled_edges_negative], dim=0
        )

        article_features = torch.empty(
            size=(
                len(all_sampled_edges),
                self.graph["article"].x[self.edges[0][0]].shape[0],
            )
        )
        for i, article_id in enumerate(all_sampled_edges.to(torch.long)):
            article_features[i] = self.graph["article"].x[article_id]

        """ Edges """
        # Define the whole graph and the subgraph
        all_edges = self.graph[edge_key].edge_index
        subgraph_edges = torch.tensor(self.edges[idx])

        # Sample positive edges from subgraph
        subgraph_sample_positive = subgraph_edges[:1]

        # Sample negative edges from the whole graph, filtering out subgraph edges (positive edges)
        sampled_edges_negative = self.__get_negative_edges(
            filter_ids=subgraph_edges,
            potential_edges=all_edges,
            num_negative_edges=len(subgraph_sample_positive),
        )

        # Expand flat edge list with user's id to have shape [2, num_nodes]
        id_tensor = torch.tensor([idx])
        all_sampled_edges = torch.stack(
            [
                id_tensor.repeat(len(all_sampled_edges)),
                all_sampled_edges,
            ],
            dim=0,
        )

        # Prepare identifier of labels
        labels = torch.cat(
            [
                torch.ones(subgraph_sample_positive.shape[0]),
                torch.zeros(sampled_edges_negative.shape[0]),
            ],
            dim=0,
        )

        """ Create Data """
        data = HeteroData()
        data["customer"].x = torch.unsqueeze(user_features, dim=0)
        data["article"].x = article_features

        # Add original directional edges
        data[edge_key].edge_index = subgraph_edges
        data[edge_key].edge_label_index = all_sampled_edges
        data[edge_key].edge_label = labels

        # Add reverse edges
        reverse_key = torch.LongTensor([1, 0])
        data[rev_edge_key].edge_index = subgraph_edges[reverse_key]
        data[rev_edge_key].edge_label_index = all_sampled_edges[reverse_key]
        data[rev_edge_key].edge_label = labels
        return data

        # all_edges = self.graph[edge_key].edge_index
        # all_edges_mapped = all_edges.clone()
        # all_edges_mapped[0] = self.__remap_indexes_to_zero(all_edges_mapped[0])
        # all_edges_mapped[1] = self.__remap_indexes_to_zero(all_edges_mapped[1])

        # """ Prepare features and edges """
        # user_features = self.graph["customer"].x[idx]
        # article_features = torch.empty(
        #     size=(
        #         len(all_edges_mapped),
        #         self.graph["article"].x[self.edges[0][0]].shape[0],
        #     )
        # )
        # for i in range(len(all_edges_mapped)):
        #     article_features[i] = self.graph["article"].x[i]

        # # Get positive and negative edges
        # positive_edges_flat = torch.tensor(self.edges[idx])
        # positive_edges_flat = self.__remap_indexes_to_zero(
        #     positive_edges_flat, buckets=all_edges_mapped[1]
        # )
        # negative_edges_flat = self.__get_negative_edges(
        #     filter_ids=positive_edges_flat,
        #     potential_edges=all_edges_mapped,
        #     num_negative_edges=len(positive_edges_flat),
        # )

        # # Add customer id to the tensor
        # id_tensor = torch.tensor([idx])
        # positive_edges = torch.stack(
        #     [id_tensor.repeat(len(positive_edges_flat)), positive_edges_flat],
        #     dim=0,
        # )
        # negative_edges = torch.stack(
        #     [id_tensor.repeat(len(negative_edges_flat)), negative_edges_flat]
        # )
        # pos_neg_edges = torch.cat([positive_edges, negative_edges], dim=1)

        # # Create labels that identify negative from positive edges
        # labels = torch.cat(
        #     [torch.ones(positive_edges.shape[1]), torch.zeros(negative_edges.shape[1])],
        #     dim=0,
        # )

        # """ Create Data """
        # data = HeteroData()
        # data["customer"].x = torch.unsqueeze(user_features, dim=0)
        # data["article"].x = article_features

        # # Add original directional edges
        # data[edge_key].edge_index = all_edges_mapped
        # data[edge_key].edge_label_index = pos_neg_edges
        # data[edge_key].edge_label = labels

        # # Add reverse edges
        # reverse_key = torch.LongTensor([1, 0])
        # data[rev_edge_key].edge_index = all_edges_mapped[reverse_key]
        # data[rev_edge_key].edge_label_index = pos_neg_edges[reverse_key]
        # data[rev_edge_key].edge_label = labels
        # return data

    def __get_negative_edges(
        self,
        filter_ids: Tensor,
        potential_edges: Tensor,
        num_negative_edges: int = 10,
    ) -> Tensor:
        # Get the biggest value available in articles (potential edges to sample from)
        id_max = torch.max(potential_edges, dim=1)[0][1]

        # Create list of potential negative edges, filter out positive edges
        combined = torch.cat((torch.range(start=0, end=id_max), filter_ids))
        uniques, counts = combined.unique(return_counts=True)
        difference = uniques[counts == 1]

        # Randomly sample negative edges
        negative_edges = difference[torch.randperm(difference.nelement())][
            :num_negative_edges
        ]

        return negative_edges

    def __remap_indexes_to_zero(
        self, all_edges: Tensor, buckets: Optional[Tensor] = None
    ) -> Tensor:
        all_edges_copy = all_edges.clone()

        # If there are no buckets it should remap on itself
        if buckets is None:
            buckets = torch.unique(all_edges_copy)

        return torch.bucketize(all_edges_copy, buckets)
