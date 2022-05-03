from typing import Literal
from utils.flatten import flatten
from .type import Matcher
import torch
from torch import Tensor


class UsersWithCommonPurchasesMatcher(Matcher):
    def __init__(self, k: int, suffix):  ##: Literal["train", "test", "val"]):
        self.user_to_articles = torch.load(f"data/derived/edges_{suffix}.pt")
        self.article_to_users = torch.load(f"data/derived/rev_edges_{suffix}.pt")
        self.k = k

    def get_matches(self, user_id: int) -> torch.Tensor:
        articles_purchased = self.user_to_articles[user_id]
        users_with_same_articles = flatten(
            [self.article_to_users[article] for article in articles_purchased]
        )
        articles_purchased_by_common_users = torch.cat(
            [
                torch.as_tensor(self.user_to_articles[user])
                for user in users_with_same_articles
            ],
            dim=0,
        )
        return articles_purchased_by_common_users[: self.k]


def load_bucketized_() -> Tensor:
    item_embeddings: Tensor = torch.load("data/derived/items_emb_final_lightgcn.pt")
    user_embeddings: Tensor = torch.load("data/derived/users_emb_final_lightgcn.pt")
    return item_embeddings @ user_embeddings.T
