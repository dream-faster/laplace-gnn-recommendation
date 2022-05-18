from .type import Matcher
import torch as t


class LightGCNMatcher(Matcher):
    def __init__(self, k: int):  # : Literal["train", "test", "val"]
        self.top_articles_per_user = t.load("data/derived/lightgcn_output.pt")
        self.k = k

    def get_matches(self, user_id: int) -> t.Tensor:
        return self.top_articles_per_user[user_id][: self.k]
