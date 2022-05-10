from abc import ABC
import torch
from typing import List


class GraphQueryServer(ABC):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def get_item(self, idx: int) -> List[int]:
        raise NotImplementedError
