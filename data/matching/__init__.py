from .lightgcn import LightGCNMatcher
from .users_with_common_purchases import UsersWithCommonItemsMatcher
from .fashion.users_same_location import UsersSameLocationMatcher
from .fashion.popular_items import PopularItemsMatcher
from typing import List
from .type import Matcher


def get_matchers(
    dataset_type: str, split_name: str, candidate_pool_size: int
) -> List[Matcher]:
    if dataset_type == "movielens":
        return [
            UsersWithCommonItemsMatcher(candidate_pool_size, split_name),
        ]
    elif dataset_type == "fashion":
        return [
            # LightGCNMatcher(config.candidate_pool_size),
            PopularItemsMatcher(candidate_pool_size),
            # UsersSameLocationMatcher(config.candidate_pool_size, split_name),
            UsersWithCommonItemsMatcher(candidate_pool_size, split_name),
        ]
    else:
        raise ValueError("Unknown matchers type: {}".format(dataset_type))
