from .type import Matcher
import torch


class UsersSameLocationMatcher(Matcher):
    def __init__(self, k: int):
        self.customers_per_location = torch.load(
            "data/derived/customers_per_location.pt"
        )
        self.location_for_user = torch.load("data/derived/location_for_user.pt")
        self.k = k

    def get_matches(self, user_id: int) -> torch.Tensor:
        location = self.location_for_user[user_id]
        return torch.Tensor(self.customers_per_location[location][: self.k])
