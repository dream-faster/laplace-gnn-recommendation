from numpy import dtype
from .type import Matcher
import torch

class PopularItemsMatcher(Matcher):
    def __init__(self, k: int):
        self.location_for_user = torch.Tensor(
            torch.load("data/derived/most_popular_products.pt")
        ).to(torch.long)
        self.k = k

    def get_matches(self, user_id: int) -> torch.Tensor:
        return self.location_for_user[: self.k]
