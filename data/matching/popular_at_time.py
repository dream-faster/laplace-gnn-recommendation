from numpy import dtype
from .type import Matcher
import torch

# from typing import Literal


class PopularAtTime(Matcher):
    def __init__(self, k: int, suffix):  # : Literal["train", "test", "val"]
        self.customers_per_location = torch.load(
            "data/derived/customers_per_location.pt"
        )
        self.location_for_user = torch.load("data/derived/location_for_user.pt")
        self.user_to_articles = torch.load(f"data/derived/edges_{suffix}.pt")
        self.k = k

    def get_matches(self, user_id: int) -> torch.Tensor:
        location = self.location_for_user[user_id]
        customers_at_location = self.customers_per_location[location]

        return torch.cat(
            [
                torch.as_tensor(self.user_to_articles[user])
                for user in customers_at_location
            ],
            dim=0,
        )[: self.k]
