from abc import ABC
import torch


class GraphQueryServer(ABC):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def get_item(self, idx: int) -> torch.Tensor:
        raise NotImplementedError
