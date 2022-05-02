from abc import ABC
import torch

class Matcher(ABC):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError
        
    def get_matches(self, user_id: int) -> torch.Tensor:
        raise NotImplementedError