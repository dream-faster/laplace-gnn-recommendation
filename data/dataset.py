import torch
import math
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch import Tensor
from typing import Union, Optional, List
from .matching.type import Matcher
from utils.constants import Constants
from config import DataLoaderConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphDataset(InMemoryDataset):
    def __init__(
        self,
        config: DataLoaderConfig,
        edge_path: str,
        graph_path: str,
        train: bool,
        matchers: Optional[List[Matcher]] = None,
    ):
        self.edges = torch.load(edge_path)
        self.graph = torch.load(graph_path)
        self.matchers = matchers
        self.config = config
        self.train = train

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, idx: int) -> Union[Data, HeteroData]:
        """Create Edges"""
        # Define the whole graph and the subgraph
        all_edges = self.graph[Constants.edge_key].edge_index
        subgraph_edges = torch.tensor(self.edges[idx])

        samp_cut = max(
            1, math.floor(len(subgraph_edges) * self.config.positive_edges_ratio)
        )

        # Sample positive edges from subgraph
        subgraph_sample_positive = subgraph_edges[
            torch.randint(low=0, high=len(self.edges[idx]), size=(samp_cut,))
        ]

        if self.train:
            # Randomly select from the whole graph
            sampled_edges_negative = get_negative_edges_random(
                subgraph_edges_to_filter=subgraph_edges,
                all_edges=all_edges,
                num_negative_edges=int(
                    self.config.negative_edges_ratio * len(subgraph_sample_positive)
                ),
            )
        else:
            assert self.matchers is not None, "Must provide matchers for test"
            # Select according to a heuristic (eg.: lightgcn scores)
            candidates = torch.cat(
                [matcher.get_matches(idx) for matcher in self.matchers],
                dim=0,
            )
            # but never add positive edges
            sampled_edges_negative = only_items_with_count_one(
                torch.cat((candidates.unique(), subgraph_edges), dim=0)
            )

        all_touched_edges = torch.cat([subgraph_edges, sampled_edges_negative], dim=0)

        """ Node Features """
        # Prepare user features
        user_features = self.graph[Constants.node_user].x[idx]

        # Prepare connected article features
        article_features = torch.empty(
            size=(
                len(all_touched_edges),
                self.graph[Constants.node_item].x[self.edges[0][0]].shape[0],
            )
        )
        for i, article_id in enumerate(all_touched_edges):
            article_features[i] = self.graph[Constants.node_item].x[article_id]

        """ Remap and Prepare Edges """
        # Remap IDs
        subgraph_edges_remapped = remap_indexes_to_zero(
            subgraph_edges, buckets=torch.unique(subgraph_edges)
        )
        subgraph_sample_positive_remapped = remap_indexes_to_zero(
            subgraph_sample_positive
        )
        sampled_edges_negative_remapped = remap_indexes_to_zero(sampled_edges_negative)
        #
        sampled_edges_negative_remapped += len(subgraph_edges)

        all_sampled_edges_remapped = torch.cat(
            [subgraph_sample_positive_remapped, sampled_edges_negative_remapped], dim=0
        )

        # Expand flat edge list with user's id to have shape [2, num_nodes]
        id_tensor = torch.tensor([0])
        all_sampled_edges_remapped = torch.stack(
            [
                id_tensor.repeat(len(all_sampled_edges_remapped)),
                all_sampled_edges_remapped,
            ],
            dim=0,
        )
        subgraph_edges_remapped = torch.stack(
            [
                id_tensor.repeat(len(subgraph_edges_remapped)),
                subgraph_edges_remapped,
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
        data[Constants.node_user].x = torch.unsqueeze(user_features, dim=0)
        data[Constants.node_item].x = article_features

        # Add original directional edges
        data[Constants.edge_key].edge_index = subgraph_edges_remapped
        data[Constants.edge_key].edge_label_index = all_sampled_edges_remapped
        data[Constants.edge_key].edge_label = labels

        # Add reverse edges
        reverse_key = torch.LongTensor([1, 0])
        data[Constants.rev_edge_key].edge_index = subgraph_edges_remapped[reverse_key]
        data[Constants.rev_edge_key].edge_label_index = all_sampled_edges_remapped[
            reverse_key
        ]
        data[Constants.rev_edge_key].edge_label = labels
        return data


def only_items_with_count_one(input: torch.Tensor) -> torch.Tensor:
    uniques, counts = input.unique(return_counts=True)
    return uniques[counts == 1]


def get_negative_edges_random(
    subgraph_edges_to_filter: Tensor,
    all_edges: Tensor,
    num_negative_edges: int,
) -> Tensor:

    # Get the biggest value available in articles (potential edges to sample from)
    id_max = torch.max(all_edges, dim=1)[0][1]

    if all_edges.shape[1] / num_negative_edges > 100:
        # If the number of edges is high, it is unlikely we get a positive edge, no need for expensive filter operations
        return torch.randint(low=0, high=id_max.item(), size=(num_negative_edges,))

    else:
        # Create list of potential negative edges, filter out positive edges
        only_negative_edges = only_items_with_count_one(
            torch.cat(
                torch.range(start=0, end=id_max, dtype=torch.int64),
                subgraph_edges_to_filter,
            )
        )

        # Randomly sample negative edges
        negative_edges = only_negative_edges[
            torch.randperm(only_negative_edges.nelement())
        ][:num_negative_edges]

        return negative_edges


def remap_indexes_to_zero(
    all_edges: Tensor, buckets: Optional[Tensor] = None
) -> Tensor:
    # If there are no buckets it should remap on itself
    if buckets is None:
        buckets = torch.unique(all_edges)

    return torch.bucketize(all_edges, buckets)
