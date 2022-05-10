# from typing import Literal
from data.query_server.queries import ArticleQueryServer, UserQueryServer
from utils.flatten import flatten
from .type import Matcher
import torch
from torch import Tensor


class ArticlesWithCommonUsersMatcher(Matcher):
    def __init__(
        self,
        k: int,
    ):
        self.k = k

    def get_matches(self, user_id: int) -> torch.Tensor:
        articles_purchased = self.users.get_item(user_id)
        users_with_same_articles = flatten(
            [self.articles.get_item(article) for article in articles_purchased]
        )
        articles_purchased_by_common_users = torch.cat(
            [
                torch.as_tensor(self.users.get_item(user), dtype=torch.long)
                for user in users_with_same_articles
            ],
            dim=0,
        )
        return articles_purchased_by_common_users[: self.k]
