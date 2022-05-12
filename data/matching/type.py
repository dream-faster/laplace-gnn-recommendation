from abc import ABC
import torch as t


class Matcher(ABC):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def get_matches(self, user_id: int) -> t.Tensor:
        raise NotImplementedError
