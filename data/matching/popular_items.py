from numpy import dtype
from .type import Matcher
import torch as t


class PopularItemsMatcher(Matcher):
    def __init__(self, k: int):
        self.popular_items = t.Tensor(
            t.load("data/derived/most_popular_products.pt")
        ).to(t.long)
        self.k = k

    def get_matches(self, user_id: int) -> t.Tensor:
        return self.popular_items[: self.k]
