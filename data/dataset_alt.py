import torch as t
import math
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch import Tensor
from typing import Union, Optional, List
from .matching.type import Matcher
from utils.constants import Constants
from config import Config
from typing import Tuple
from utils.tensor import difference_1d


class GraphDataset(InMemoryDataset):
    def __init__(
        self,
        config: Config,
        graph_path: str,
        users_adj_list: str,
        articles_adj_list: str,
        train: bool,
        matchers: Optional[List[Matcher]] = None,
        randomization: bool = True,
    ):
        self.edges = t.load(users_adj_list)
        self.graph = t.load(graph_path)
        self.article_edges = t.load(articles_adj_list)
        self.matchers = matchers
        self.config = config
        self.train = train
        self.randomization = randomization

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, idx: int) -> Union[Data, HeteroData]:
        """Create Edges"""
        num_hops = 2

        # Define the whole graph and the subgraph
        all_edges = self.graph[Constants.edge_key].edge_index

        # Add first user to user_to_check
        users_to_check = t.tensor([idx])
        old_users_to_check = t.tensor([idx])
        subgraph_edges_list = []

        for i in range(num_hops):
            """For origin user also sample positive edges and negative edges"""
            if i == 0:
                (
                    subgraph_edges,
                    subgraph_sample_positive,
                    sampled_edges_negative,
                    labels,
                ) = self.first_user(idx, all_edges)

                subgraph_edges_list.append(subgraph_edges)
                continue

            """ Define new subset of users"""
            connected_articles = t.unique(
                t.tensor([a for _id in users_to_check for a in self.edges[_id.item()]])
            )

            # Keep track of what user we already sampled
            old_users_to_check = t.concat(
                (old_users_to_check, t.clone(users_to_check)), dim=0
            )

            # Get new users by looking at all the edges of the articles we were connected to
            users_to_check = t.tensor(
                [
                    a
                    for node_id in connected_articles
                    for a in self.article_edges[node_id.item()]
                ]
            )

            # Filter out users we already sampled
            users_to_check = difference_1d(
                users_to_check, old_users_to_check, assume_unique=True
            )

            """ Loop through and add edges to the subgraph """
            for user_id in users_to_check:
                subgraph_edges = self.single_user(user_id.item())
                subgraph_edges_list.append(subgraph_edges)

        # The entire subgraph with positive edges (negative edges excluded)
        subgraph_edges = t.concat(subgraph_edges_list, dim=1)

        """ Get Features """
        all_touched_edges = t.concat([subgraph_edges, sampled_edges_negative], dim=1)

        all_customer_ids = t.unique(all_touched_edges[0])
        all_article_ids = t.unique(all_touched_edges[1])
        user_features, article_features = self.get_features(
            all_customer_ids=all_customer_ids, all_article_ids=all_article_ids
        )

        """ Remap Edges """
        # Remap IDs
        buckets_customer = t.unique(subgraph_edges[0])
        buckets_articles = t.unique(subgraph_edges[1])
        (
            subgraph_edges_remapped,
            subgraph_sample_positive_remapped,
            sampled_edges_negative_remapped,
        ) = self.remap_edges(
            subgraph_edges,
            subgraph_sample_positive,
            sampled_edges_negative,
            all_touched_edges,
            buckets_customers=buckets_customer,
            buckets_articles=buckets_articles,
        )

        # The subgraph and the sampled graph together (with negative and positive samples)
        all_sampled_edges_remapped = t.concat(
            [subgraph_sample_positive_remapped, sampled_edges_negative_remapped], dim=1
        )

        """ Create Data """
        data = HeteroData()
        if len(user_features.shape) == 1:
            user_features = t.unsqueeze(user_features, dim=0)
        data[Constants.node_user].x = user_features
        data[Constants.node_item].x = article_features

        # Add original directional edges
        data[Constants.edge_key].edge_index = subgraph_edges_remapped
        data[Constants.edge_key].edge_label_index = all_sampled_edges_remapped
        data[Constants.edge_key].edge_label = labels

        # Add reverse edges
        reverse_key = t.LongTensor([1, 0])
        data[Constants.rev_edge_key].edge_index = subgraph_edges_remapped[reverse_key]
        data[Constants.rev_edge_key].edge_label_index = all_sampled_edges_remapped[
            reverse_key
        ]
        data[Constants.rev_edge_key].edge_label = labels
        return data

    def single_user(self, idx: int) -> Tensor:
        subgraph_edges_flat = t.tensor(self.edges[idx])
        id_tensor = t.tensor([idx])
        subgraph_edges = t.stack(
            [
                id_tensor.repeat(len(subgraph_edges_flat)),
                subgraph_edges_flat,
            ],
            dim=0,
        )
        return subgraph_edges

    def first_user(self, idx: int, all_edges: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        subgraph_edges = t.tensor(self.edges[idx])

        samp_cut = max(
            1, math.floor(len(subgraph_edges) * self.config.positive_edges_ratio)
        )

        # Sample positive edges from subgraph
        if self.randomization:
            random_integers = t.randint(
                low=0, high=len(self.edges[idx]), size=(samp_cut,)
            )
        else:
            random_integers = t.tensor([0, t.max(subgraph_edges, dim=0)[1].item()])

        subgraph_sample_positive = subgraph_edges[random_integers]

        if self.train:
            # Randomly select from the whole graph
            sampled_edges_negative = get_negative_edges_random(
                subgraph_edges_to_filter=subgraph_edges,
                all_edges=all_edges,
                num_negative_edges=int(
                    self.config.negative_edges_ratio * len(subgraph_sample_positive)
                ),
                randomization=self.randomization,
            )
        else:
            assert self.matchers is not None, "Must provide matchers for test"
            # Select according to a heuristic (eg.: lightgcn scores)
            candidates = t.cat(
                [matcher.get_matches(idx) for matcher in self.matchers],
                dim=0,
            )
            # but never add positive edges
            sampled_edges_negative = only_items_with_count_one(
                t.cat([candidates.unique(), subgraph_edges], dim=0)
            )

        """ Remap and Prepare Edges """
        # Expand flat edge list with user's id to have shape [2, num_nodes]
        id_tensor = t.tensor([0])

        subgraph_sample_positive = t.stack(
            [
                id_tensor.repeat(len(subgraph_sample_positive)),
                subgraph_sample_positive,
            ],
            dim=0,
        )
        sampled_edges_negative = t.stack(
            [
                id_tensor.repeat(len(sampled_edges_negative)),
                sampled_edges_negative,
            ],
            dim=0,
        )

        subgraph_edges = t.stack(
            [
                id_tensor.repeat(len(subgraph_edges)),
                subgraph_edges,
            ],
            dim=0,
        )

        # Prepare identifier of labels
        labels = t.cat(
            [
                t.ones(subgraph_sample_positive.shape[1]),
                t.zeros(sampled_edges_negative.shape[1]),
            ],
            dim=0,
        )
        return (
            subgraph_edges,
            subgraph_sample_positive,
            sampled_edges_negative,
            labels,
        )

    def get_features(
        self, all_customer_ids: Tensor, all_article_ids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Node Features"""
        # Prepare user features
        user_features = self.graph[Constants.node_user].x[all_customer_ids]

        # Prepare connected article features
        article_features = self.graph[Constants.node_item].x[all_article_ids]

        return user_features, article_features

    def remap_edges(
        self,
        subgraph_edges: Tensor,
        subgraph_sample_positive: Tensor,
        sampled_edges_negative: Tensor,
        all_touched_edges: Tensor,
        buckets_customers: Tensor,
        buckets_articles: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Remap and Prepare Edges"""

        """ Subgraph Edges """
        subgraph_edges[0] = remap_indexes_to_zero(
            subgraph_edges[0], buckets=buckets_customers
        )
        subgraph_edges[1] = remap_indexes_to_zero(
            subgraph_edges[1], buckets=buckets_articles
        )

        """ Positive Sampled Edges """
        subgraph_sample_positive[0] = remap_indexes_to_zero(
            subgraph_sample_positive[0], buckets=buckets_customers
        )
        subgraph_sample_positive[1] = remap_indexes_to_zero(
            subgraph_sample_positive[1], buckets=buckets_articles
        )

        """ Negative Edges """
        # All negative sampled edges are connected to the root user
        id_tensor = t.tensor([0])
        sampled_edges_negative[0] = id_tensor.repeat(len(sampled_edges_negative[0]))

        # Remap negative edges to start from zero
        all_touched_edges_ids = t.unique(all_touched_edges[1])

        sampled_edges_negative[1] = remap_indexes_to_zero(
            sampled_edges_negative[1], buckets=all_touched_edges_ids
        )

        return subgraph_edges, subgraph_sample_positive, sampled_edges_negative


def only_items_with_count_one(input: t.Tensor) -> t.Tensor:
    uniques, counts = input.unique(return_counts=True)
    return uniques[counts == 1]


def get_negative_edges_random(
    subgraph_edges_to_filter: Tensor,
    all_edges: Tensor,
    num_negative_edges: int,
    randomization: bool,
) -> Tensor:

    # Get the biggest value available in articles (potential edges to sample from)
    id_max = t.max(all_edges, dim=1)[0][1]

    if all_edges.shape[1] / num_negative_edges > 100:
        # If the number of edges is high, it is unlikely we get a positive edge, no need for expensive filter operations
        if randomization:
            random_integers = t.randint(
                low=0, high=id_max.item(), size=(num_negative_edges,)
            )
        else:
            random_integers = t.tensor([id_max.item()])

        return random_integers

    else:
        # Create list of potential negative edges, filter out positive edges
        only_negative_edges = only_items_with_count_one(
            t.cat(
                (
                    t.arange(start=0, end=id_max + 1, dtype=t.int64),
                    subgraph_edges_to_filter,
                ),
                dim=0,
            )
        )

        # Randomly sample negative edges
        if randomization:
            random_integers = t.randperm(only_negative_edges.nelement())
            negative_edges = only_negative_edges[random_integers][:num_negative_edges]
        else:
            negative_edges = t.tensor([id_max.item()])

        return negative_edges


def remap_indexes_to_zero(all_edges: Tensor, buckets: Tensor) -> Tensor:
    return t.bucketize(all_edges, buckets)
