import torch as t
from torch import Tensor
from typing import List, Union
import torch.nn.functional as F
import numpy as np


def intersection_1d(t1: Tensor, t2: Tensor) -> Tensor:
    # This can be quite costly, seek for alternative implementations before using it in prod
    combined = t.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    intersection = uniques[counts > 1]
    return intersection


def difference_1d(a: Tensor, b: Tensor, assume_unique: bool) -> Tensor:
    r"""Returns the elements of A without the elements of B 1D"""
    diff = np.setdiff1d(
        a.detach().numpy(), b.detach().numpy(), assume_unique=assume_unique
    )
    return t.tensor(diff)


def padded_stack(
    tensors: List[t.Tensor],
    side: str = "right",
    mode: str = "constant",
    value: Union[int, float] = 0,
) -> t.Tensor:
    """
    Stack tensors along first dimension and pad them along last dimension to ensure their size is equal.

    Args:
        tensors (List[t.Tensor]): list of tensors to stack
        side (str): side on which to pad - "left" or "right". Defaults to "right".
        mode (str): 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        value (Union[int, float]): value to use for constant padding

    Returns:
        t.Tensor: stacked tensor
    """
    full_size = max([x.size(-1) for x in tensors])

    def make_padding(pad):
        if side == "left":
            return (pad, 0)
        elif side == "right":
            return (0, pad)
        else:
            raise ValueError(f"side for padding '{side}' is unknown")

    out = t.stack(
        [
            F.pad(x, make_padding(full_size - x.size(-1)), mode=mode, value=value)
            if full_size - x.size(-1) > 0
            else x
            for x in tensors
        ],
        dim=0,
    )
    return out


def check_edge_index_flat_unique(edge_index: t.Tensor) -> t.Tensor:
    return t.tensor(list(set(tuple(pair) for pair in edge_index.t().tolist())))
