import torch
import math
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch import Tensor
from typing import Tuple, Union, Optional, List
from .matching.type import Matcher
from utils.constants import Constants
from config import Config
from utils.flatten import flatten

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphDataset(InMemoryDataset):
    def __init__(
        self,
        config: Config,
        graph_path: str,
        users_adj_list: str,
        articles_adj_list: str,
        train: bool,
        matchers: Optional[List[Matcher]] = None,
    ):

        self.graph = torch.load(graph_path)
        self.articles = torch.load(users_adj_list)
        self.users = torch.load(articles_adj_list)
        self.matchers = matchers
        self.config = config
        self.train = train

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Union[Data, HeteroData]:
        """Create Edges"""
        all_edges = self.graph[Constants.edge_key].edge_index
        positive_article_indices = torch.as_tensor(
            self.users[idx], dtype=torch.long
        )  # all the positive target indices for the current user
        positive_article_edges = create_edges_from_target_indices(
            idx, positive_article_indices
        )

        # Sample positive edges from subgraph (amount defined in config.positive_edges_ratio)
        samp_cut = max(
            1,
            math.floor(
                len(positive_article_indices) * self.config.positive_edges_ratio
            ),
        )
        sampled_positive_article_indices = positive_article_indices[
            torch.randint(low=0, high=len(positive_article_indices), size=(samp_cut,))
        ]
        sampled_positive_article_edges = create_edges_from_target_indices(
            idx, sampled_positive_article_indices
        )

        if self.train:
            # Randomly select from the whole graph
            sampled_negative_article_edges = create_edges_from_target_indices(
                idx,
                get_negative_edges_random(
                    subgraph_edges_to_filter=sampled_positive_article_indices,
                    all_edges=all_edges,
                    num_negative_edges=int(self.config.negative_edges_ratio * samp_cut),
                ),
            )
        else:
            assert self.matchers is not None, "Must provide matchers for test"
            # Select according to a heuristic (eg.: lightgcn scores)
            candidates = torch.cat(
                [matcher.get_matches(idx) for matcher in self.matchers],
                dim=0,
            ).unique()
            # but never add positive edges
            sampled_negative_article_edges = create_edges_from_target_indices(
                idx,
                only_items_with_count_one(
                    torch.cat([candidates, positive_article_indices], dim=0)
                ),
            )

        n_hop_edges = fetch_n_hop_neighbourhood(
            self.config.num_neighbors_it, idx, self.users, self.articles
        )

        all_touched_edges = torch.cat(
            [
                positive_article_edges,
                sampled_negative_article_edges,
                n_hop_edges,
            ],
            dim=1,
        )

        all_subgraph_edges = torch.cat(
            [
                positive_article_edges,
                n_hop_edges,
            ],
            dim=1,
        )

        """ Node Features """
        user_buckets = torch.unique(all_touched_edges[0], sorted=True)
        article_buckets = torch.unique(all_touched_edges[1], sorted=True)

        user_features = self.graph[Constants.node_user].x[user_buckets]
        article_features = self.graph[Constants.node_item].x[article_buckets]

        """ Remap and Prepare Edges """
        all_subgraph_edges = remap_edges_to_start_from_zero(
            all_subgraph_edges, user_buckets, article_buckets
        )
        all_sampled_edges = remap_edges_to_start_from_zero(
            torch.cat(
                [sampled_positive_article_edges, sampled_negative_article_edges], dim=1
            ),
            user_buckets,
            article_buckets,
        )

        # Prepare identifier of labels
        labels = torch.cat(
            [
                torch.ones(sampled_positive_article_edges.shape[1]),
                torch.zeros(sampled_negative_article_edges.shape[1]),
            ],
            dim=0,
        )

        all_sampled_edges, labels = shuffle_edges_and_labels(all_sampled_edges, labels)

        """ Create Data """
        data = HeteroData()
        data[Constants.node_user].x = user_features
        data[Constants.node_item].x = article_features

        # Add original directional edges
        data[Constants.edge_key].edge_index = all_subgraph_edges
        data[Constants.edge_key].edge_label_index = all_sampled_edges
        data[Constants.edge_key].edge_label = labels

        # Add reverse edges
        reverse_key = torch.LongTensor([1, 0])
        data[Constants.rev_edge_key].edge_index = all_subgraph_edges[reverse_key]
        data[Constants.rev_edge_key].edge_label_index = all_sampled_edges[reverse_key]
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
                (
                    torch.arange(start=0, end=id_max, dtype=torch.int64),
                    subgraph_edges_to_filter,
                ),
                dim=0,
            )
        )

        # Randomly sample negative edges
        negative_edges = only_negative_edges[
            torch.randperm(only_negative_edges.nelement())
        ][:num_negative_edges]

        return negative_edges


def remap_edges_to_start_from_zero(
    edges: Tensor, buckets_1st_dim: Tensor, buckets_2nd_dim: Tensor
) -> Tensor:
    return torch.stack(
        (
            torch.bucketize(edges[0], buckets_1st_dim),
            torch.bucketize(edges[1], buckets_2nd_dim),
        )
    )


def create_edges_from_target_indices(
    source_index: int, target_indices: Tensor
) -> Tensor:
    """Expand target indices list with user's id to have shape [2, num_nodes]"""

    return torch.stack(
        [
            torch.Tensor([source_index])
            .to(dtype=torch.long)
            .repeat(len(target_indices)),
            torch.as_tensor(target_indices, dtype=torch.long),
        ],
        dim=0,
    )


def fetch_n_hop_neighbourhood(
    n: int, user_id: int, users: dict, articles: dict
) -> torch.Tensor:
    """Returns the edges from the n-hop neighbourhood of the user, without the direct links for the same user"""
    accum_edges = torch.Tensor([[], []]).to(dtype=torch.long)
    users_explored = set([])
    users_queue = set([user_id])
    articles_queue = []

    for i in range(0, n):
        new_articles_and_edges = [
            create_neighbouring_article_edges(user, users) for user in users_queue
        ]
        users_explored = users_explored | users_queue
        if len(new_articles_and_edges) == 0:
            break
        new_articles = flatten([x[0] for x in new_articles_and_edges])

        if i != 0:
            new_edges = torch.cat([x[1] for x in new_articles_and_edges], dim=1)
            accum_edges = torch.cat([accum_edges, new_edges], dim=1)

        articles_queue.extend(new_articles)
        new_users = (
            set(flatten([articles[article] for article in articles_queue]))
            - users_explored
        )  # remove the intersection between the two sets, so we only explore a user once
        users_queue = new_users

    return accum_edges


def create_neighbouring_article_edges(
    user_id: int, users: dict
) -> Tuple[List[int], Tensor]:
    """Fetch neighbouring articles for a user, returns the article ids (list[int]) & edges (Tensor)"""
    articles_purchased = users[user_id]
    edges_to_articles = create_edges_from_target_indices(
        user_id, torch.as_tensor(articles_purchased, dtype=torch.long)
    )
    return articles_purchased, edges_to_articles


def shuffle_edges_and_labels(edges: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
    new_edge_order = torch.randperm(edges.size(1))
    return (edges[:, new_edge_order], labels[new_edge_order])
