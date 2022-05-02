import torch
import math
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch import Tensor
from typing import Union, Optional


def get_negative_edges_random(
    filter_ids: Tensor,
    potential_edges: Tensor,
    num_negative_edges: int = 10,
) -> Tensor:
    # Get the biggest value available in articles (potential edges to sample from)
    id_max = torch.max(potential_edges, dim=1)[0][1]

    # Create list of potential negative edges, filter out positive edges
    combined = torch.cat(
        (torch.range(start=0, end=id_max, dtype=torch.int64), filter_ids)
    )
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]

    # Randomly sample negative edges
    negative_edges = difference[torch.randperm(difference.nelement())][
        :num_negative_edges
    ]

    return negative_edges


def remap_indexes_to_zero(
    all_edges: Tensor, buckets: Optional[Tensor] = None
) -> Tensor:
    # If there are no buckets it should remap on itself
    if buckets is None:
        buckets = torch.unique(all_edges)

    return torch.bucketize(all_edges, buckets)


def get_negative_edges_candidate_selection(
    user_id: int, scores: Tensor, k: int
) -> Tensor:
    candidate_scores_all: Tensor = scores[user_id]
    candidate_scores, candidate_ids = torch.topk(candidate_scores_all, k=k)

    return candidate_ids


def get_scores() -> Tensor:
    item_embeddings: Tensor = torch.load("data/derived/items_emb_final_lightgcn.pt")
    user_embeddings: Tensor = torch.load("data/derived/users_emb_final_lightgcn.pt")

    return item_embeddings @ user_embeddings.T


class GraphDataset(InMemoryDataset):
    def __init__(
        self, edge_dir: str, graph_dir: str, eval: bool, candidate_pool_size: int
    ):
        self.edges = torch.load(edge_dir)
        self.graph = torch.load(graph_dir)
        self.eval = eval
        if self.eval:
            self.lightgcn_scores = get_scores()
            self.candidate_pool_size = candidate_pool_size

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, idx: int) -> Union[Data, HeteroData]:
        edge_key = ("customer", "buys", "article")
        rev_edge_key = ("article", "rev_buys", "customer")
        cut_ratio = 0.3

        """ Create Edges """
        # Define the whole graph and the subgraph
        all_edges = self.graph[edge_key].edge_index
        subgraph_edges = torch.tensor(self.edges[idx])

        samp_cut = max(1, math.floor(len(subgraph_edges) * cut_ratio))

        # Sample positive edges from subgraph
        subgraph_sample_positive = subgraph_edges[:samp_cut]

        # Sample negative edges from the whole graph, filtering out subgraph edges (positive edges)
        if self.eval:
            # Select according to a heuristic (eg.: lightgcn scores)
            sampled_edges_negative = get_negative_edges_candidate_selection(
                idx, self.lightgcn_scores, k=self.candidate_pool_size
            )

        else:
            # Randomly select from the whole graph
            sampled_edges_negative = get_negative_edges_random(
                filter_ids=subgraph_edges,
                potential_edges=all_edges,
                num_negative_edges=len(subgraph_sample_positive),
            )

        all_touched_edges = torch.cat([subgraph_edges, sampled_edges_negative], dim=0)

        """ Node Features """
        # Prepare user features
        user_features = self.graph["customer"].x[idx]

        # Prepare connected article features
        article_features = torch.empty(
            size=(
                len(all_touched_edges),
                self.graph["article"].x[self.edges[0][0]].shape[0],
            )
        )
        for i, article_id in enumerate(all_touched_edges):
            article_features[i] = self.graph["article"].x[article_id]

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
        data["customer"].x = torch.unsqueeze(user_features, dim=0)
        data["article"].x = article_features

        # Add original directional edges
        data[edge_key].edge_index = subgraph_edges_remapped
        data[edge_key].edge_label_index = all_sampled_edges_remapped
        data[edge_key].edge_label = labels

        # Add reverse edges
        reverse_key = torch.LongTensor([1, 0])
        data[rev_edge_key].edge_index = subgraph_edges_remapped[reverse_key]
        data[rev_edge_key].edge_label_index = all_sampled_edges_remapped[reverse_key]
        data[rev_edge_key].edge_label = labels
        return data
