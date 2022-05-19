from numpy import dtype
from ..type import Matcher
import torch as t

# from typing import Literal


class UsersSameLocationMatcher(Matcher):
    def __init__(self, k: int, suffix):  # : Literal["train", "test", "val"]
        self.customers_per_location = t.load("data/derived/customers_per_location.pt")
        self.location_for_user = t.load("data/derived/location_for_user.pt")
        self.user_to_articles = t.load(f"data/derived/edges_{suffix}.pt")
        self.k = k

    def get_matches(self, user_id: int) -> t.Tensor:
        location = self.location_for_user[user_id]
        customers_at_location = self.customers_per_location[location]

        return t.cat(
            [
                t.as_tensor(self.user_to_articles[user])
                for user in customers_at_location
            ],
            dim=0,
        )[: self.k]
