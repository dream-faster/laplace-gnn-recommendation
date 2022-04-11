from torch import Tensor
from numpy import ndarray
from typing import Union


def weighted_mse_loss(pred: Tensor, target: Tensor, weight=None) -> Tensor:
    weight = 1.0 if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()
