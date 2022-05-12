from torch import Tensor
import torch as t


def intersection(t1: Tensor, t2: Tensor) -> Tensor:
    combined = t.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    intersection = uniques[counts > 1]
    return intersection


def difference(t1: Tensor, t2: Tensor) -> Tensor:
    combined = t.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    return difference
