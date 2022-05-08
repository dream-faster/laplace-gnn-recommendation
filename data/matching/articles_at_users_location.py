from numpy import dtype
from .type import Matcher
import torch
from ..query_server.queries import UserQueryServer


class ArticlesPurchasedAtUserLocationMatcher(Matcher):
    def __init__(self, k: int):
        self.customers_per_location = torch.load(
            "data/derived/customers_per_location.pt"
        )
        self.location_for_user = torch.load("data/derived/location_for_user.pt")
        self.k = k

    def get_matches(self, user_id: int) -> torch.Tensor:
        location = self.location_for_user[user_id]
        customers_at_location = self.customers_per_location[location]

        return torch.cat(
            [self.users.get_item(user) for user in customers_at_location],
            dim=0,
        )[: self.k]
