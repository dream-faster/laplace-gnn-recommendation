from .types import GraphQueryServer
import torch
from typing import Optional

K_LIMIT = 10000000


class UserQueryServer(GraphQueryServer):
    def __init__(
        self, k: Optional[int], suffix: str
    ):  ##: Literal["train", "test", "val"]):
        self.user_to_articles = torch.load(f"data/derived/edges_{suffix}.pt")
        self.k = k if k is not None else K_LIMIT

    def get_item(self, idx: int) -> torch.Tensor:
        return self.user_to_articles[idx][:self.k]

    def __len__(self):
        return len(self.user_to_articles)


class ArticleQueryServer(GraphQueryServer):
    def __init__(
        self, k: Optional[int], suffix: str
    ):  ##: Literal["train", "test", "val"]):
        self.article_to_users = torch.load(f"data/derived/rev_edges_{suffix}.pt")
        self.k = k if k is not None else K_LIMIT

    def get_item(self, idx: int) -> torch.Tensor:
        return self.article_to_users[idx][:self.k]

    def __len__(self):
        return len(self.article_to_users)
