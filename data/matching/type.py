from abc import ABC
import torch
from data.query_server.queries import ArticleQueryServer, UserQueryServer


class Matcher(ABC):

    users: UserQueryServer  # I'm sorry for the terrible OOP pattern of injecting these after initalization...
    articles: ArticleQueryServer

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def get_matches(self, user_id: int) -> torch.Tensor:
        raise NotImplementedError
