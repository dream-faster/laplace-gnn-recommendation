from numpy import dtype
import torch as t
import math
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch import Tensor
from typing import Tuple, Union, Optional, List
from .matching.type import Matcher
from utils.constants import Constants
from config import Config
from utils.flatten import flatten
import random
from data.neo4j.neo4j_database import Database
from data.neo4j.utils import get_neighborhood, get_id_map
from utils.tensor import check_edge_index_flat_unique
from collections import defaultdict


device = t.device("cuda" if t.cuda.is_available() else "cpu")


class GraphDataset(InMemoryDataset):
    def __init__(
        self,
        config: Config,
        graph_path: str,
        users_adj_list: str,
        articles_adj_list: str,
        train: bool,
        split_type: str,
        matchers: Optional[List[Matcher]] = None,
        randomization: bool = True,
        db_param: Tuple[str, str, str] = ("bolt://localhost:7687", "neo4j", "password"),
    ):

        self.graph = t.load(graph_path)
        self.articles = t.load(articles_adj_list)
        self.users = t.load(users_adj_list)
        self.matchers = matchers
        self.config = config
        self.train = train
        self.randomization = randomization
        self.db = Database(db_param[0], db_param[1], db_param[2])
        self.split_type = split_type

        self.default_edge_types = [Constants.edge_key]
        self.other_edge_types = [Constants.edge_key_extra]
        self.node_types = [Constants.node_user, Constants.node_item]

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Union[Data, HeteroData]:
        """Get Subgraph Edges, Sampled Edges and Features"""
        edge_label_index, edge_label = self.get_edge_label_index(idx)
        neighborhood = get_neighborhood(
            self.db,
            node_id=idx,
            n_neighbor=self.config.n_hop_neighbors,
            start_neighbor=1,
            split_type=self.split_type,
        )
        edge_index = self.get_edge_indexes(edge_label_index, edge_label, neighborhood)
        original_node_ids = self.get_original_node_ids(edge_index, edge_label_index)
        edge_index, edge_label_index = self.remap_indexes(
            edge_index, edge_label_index, original_node_ids
        )

        """ Create Data """
        data = HeteroData()

        for node_type in self.node_types:
            data[node_type].x = self.graph[node_type].x[original_node_ids[node_type]]

        # Add original directional edges and reverse edges
        reverse_key = t.LongTensor([1, 0])
        for edge_type in self.default_edge_types:
            data[edge_type].edge_index = edge_index[edge_type].type(t.long)
            data[edge_type].edge_label_index = edge_label_index[edge_type].type(t.long)
            data[edge_type].edge_label = edge_label[edge_type].type(t.long)

            # Reverse edges
            data["rev_" + edge_type].edge_index = edge_index[edge_type][
                reverse_key
            ].type(t.long)

        for edge_type in self.other_edge_types:
            data[edge_type].edge_index = edge_index[edge_type].type(t.long)

            # Reverse edges
            data["rev_" + edge_type].edge_index = edge_index[edge_type][
                reverse_key
            ].type(t.long)

        # data[Constants.rev_edge_key].edge_index = all_subgraph_edges[reverse_key].type(
        #     t.long
        # )
        # data[Constants.rev_edge_key].edge_label_index = all_sampled_edges[
        #     reverse_key
        # ].type(t.long)
        # data[Constants.rev_edge_key].edge_label = labels.type(t.long)

        return data

    def remap_indexes(
        self, edge_index: dict, edge_label_index: dict, original_ids: dict
    ) -> dict:
        for key, item in edge_index.items():
            edge_index[key] = remap_edges_to_start_from_zero(
                item, original_ids[key[0]], original_ids[key[2]]
            )

        for key, item in edge_label_index.items():
            edge_label_index[key] = remap_edges_to_start_from_zero(
                item, original_ids[key[0]], original_ids[key[2]]
            )

        return edge_index, edge_label_index

    def get_original_node_ids(self, edge_index: dict, edge_label_index: dict) -> dict:
        original_node_ids = defaultdict(Tensor)

        for edge_key, item in edge_index.items():
            if item.shape[0] == 0:
                continue
            original_node_ids[edge_key[0]] = t.cat(
                [original_node_ids[edge_key[0]], item[0]], dim=0
            ).type(t.long)

            original_node_ids[edge_key[2]] = t.cat(
                [original_node_ids[edge_key[2]], item[1]], dim=0
            ).type(t.long)

        for edge_key, item in edge_label_index.items():
            if item.shape[0] == 0:
                continue
            original_node_ids[edge_key[0]] = t.cat(
                [original_node_ids[edge_key[0]], item[0]], dim=0
            ).type(t.long)

            original_node_ids[edge_key[2]] = t.cat(
                [original_node_ids[edge_key[2]], item[1]], dim=0
            ).type(t.long)

        for key, item in original_node_ids.items():
            original_node_ids[key] = t.unique(item)

        return original_node_ids

    def get_edge_indexes(
        self, edge_label_index: Tensor, edge_label: Tensor, neighborhood: Tensor
    ) -> Tensor:
        """
        Combine edge indexes: neighbor (without 0-hop neighbors) and positive sampled edges
        """
        edge_index = defaultdict(list)

        for edge_type in self.default_edge_types:
            edge_index[edge_type] = t.cat(
                [
                    t.tensor(neighborhood[edge_type]),
                    edge_label_index[edge_type][:, edge_label[edge_type] == 1],
                ],
                dim=1,
            )

        for edge_type in self.other_edge_types:
            edge_index[edge_type] = t.tensor(neighborhood[edge_type])

        return edge_index

    def get_edge_label_index(self, idx: int) -> Tuple[Tensor, Tensor]:
        edge_label_index = dict()
        edge_label = dict()

        for edge_type in self.default_edge_types:
            all_edges = self.graph[edge_type].edge_index

            """ Positive Sample """
            # We will have to modify self.users to be a disctionary of self.users[idx][edge_type]
            positive_article_indices = t.as_tensor(self.users[idx], dtype=t.long)
            positive_sample = self.get_positive_sampled_edges(
                idx, positive_article_indices
            )

            """ Negative Sample """
            num_sampled_pos_edges = positive_sample.shape[0]

            if num_sampled_pos_edges <= 1:
                negative_edges_ratio = self.config.k - 1
            else:
                negative_edges_ratio = self.config.negative_edges_ratio

            negative_sample = self.get_negative_sampled_edges(
                all_edges,
                positive_sample,
                positive_article_indices,
                idx,
                negative_edges_ratio,
                num_sampled_pos_edges,
            )

            edge_label_index[edge_type] = t.cat(
                [positive_sample, negative_sample], dim=1
            )
            edge_label[edge_type] = t.cat(
                [
                    t.ones(positive_sample.shape[1]),
                    t.zeros(negative_sample.shape[1]),
                ],
                dim=0,
            )

        return edge_label_index, edge_label

    def get_positive_sampled_edges(
        self, idx: int, positive_article_indices: Tensor
    ) -> Tensor:
        # Sample positive edges from subgraph (amount defined in config.positive_edges_ratio)
        samp_cut = max(
            1,
            math.floor(
                len(positive_article_indices) * self.config.positive_edges_ratio
            ),
        )

        if self.randomization:
            random_integers = t.randint(
                low=0, high=len(positive_article_indices), size=(samp_cut,)
            )
        else:
            random_integers = t.tensor(
                [
                    t.min(positive_article_indices, dim=0)[1].item(),
                    t.max(positive_article_indices, dim=0)[1].item(),
                ]
            )

        sampled_positive_article_indices = positive_article_indices[random_integers]
        sampled_positive_article_edges = create_edges_from_target_indices(
            idx, sampled_positive_article_indices
        )
        return sampled_positive_article_edges

    def get_negative_sampled_edges(
        self,
        all_edges: Tensor,
        sampled_positive_article_indices: Tensor,
        positive_article_indices: Tensor,
        idx: int,
        negative_edges_ratio: float,
        num_sampled_pos_edges: int,
    ) -> Tensor:
        if self.train:
            # Randomly select from the whole graph
            sampled_negative_article_edges = create_edges_from_target_indices(
                idx,
                get_negative_edges_random(
                    subgraph_edges_to_filter=sampled_positive_article_indices,
                    all_edges=all_edges,
                    num_negative_edges=int(
                        negative_edges_ratio * num_sampled_pos_edges
                    ),
                    randomization=self.randomization,
                ),
            )
        else:
            assert self.matchers is not None, "Must provide matchers for test"
            # Select according to a heuristic (eg.: lightgcn scores)
            candidates = t.cat(
                [matcher.get_matches(idx) for matcher in self.matchers],
                dim=0,
            ).unique()
            # but never add positive edges
            sampled_negative_article_edges = create_edges_from_target_indices(
                idx,
                only_items_with_count_one(
                    t.cat([candidates, positive_article_indices], dim=0)
                ),
            )
        return sampled_negative_article_edges


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


def remap_edges_to_start_from_zero(
    edges: Tensor, buckets_1st_dim: Tensor, buckets_2nd_dim: Tensor
) -> Tensor:
    return t.stack(
        (
            t.bucketize(edges[0], buckets_1st_dim),
            t.bucketize(edges[1], buckets_2nd_dim),
        )
    )


def create_edges_from_target_indices(
    source_index: int, target_indices: Tensor
) -> Tensor:
    """Expand target indices list with user's id to have shape [2, num_nodes]"""

    return t.stack(
        [
            t.Tensor([source_index]).to(dtype=t.long).repeat(len(target_indices)),
            t.as_tensor(target_indices, dtype=t.long),
        ],
        dim=0,
    )


def shuffle_and_cut(array: list, n: int) -> list:
    if len(array) > n:
        return random.sample(array, n)
    else:
        return array


def create_neighbouring_article_edges(
    user_id: int, users: dict
) -> Tuple[List[int], Tensor]:
    """Fetch neighbouring articles for a user, returns the article ids (list[int]) & edges (Tensor)"""
    articles_purchased = users[user_id]
    edges_to_articles = create_edges_from_target_indices(
        user_id, t.as_tensor(articles_purchased, dtype=t.long)
    )
    return articles_purchased, edges_to_articles


def shuffle_edges_and_labels(edges: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
    new_edge_order = t.randperm(edges.size(1))
    return (edges[:, new_edge_order], labels[new_edge_order])
