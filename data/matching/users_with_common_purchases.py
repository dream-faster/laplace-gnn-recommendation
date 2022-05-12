# from typing import Literal
from utils.flatten import flatten
from .type import Matcher
import torch as t
from torch import Tensor


class UsersWithCommonPurchasesMatcher(Matcher):
    def __init__(self, k: int, suffix):  ##: Literal["train", "test", "val"]):
        self.user_to_articles = t.load(f"data/derived/edges_{suffix}.pt")
        self.article_to_users = t.load(f"data/derived/rev_edges_{suffix}.pt")
        self.k = k

    def get_matches(self, user_id: int) -> t.Tensor:
        articles_purchased = self.user_to_articles[user_id]
        users_with_same_articles = flatten(
            [self.article_to_users[article] for article in articles_purchased]
        )
        articles_purchased_by_common_users = t.cat(
            [
                t.as_tensor(self.user_to_articles[user])
                for user in users_with_same_articles
            ],
            dim=0,
        )
        return articles_purchased_by_common_users[: self.k]
