import torch as t
from torch import Tensor
from typing import List, Union
import torch.nn.functional as F


def intersection(t1: Tensor, t2: Tensor) -> Tensor:
    # This can be quite costly, seek for alternative implementations before using it in prod
    combined = t.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    intersection = uniques[counts > 1]
    return intersection


def difference(A: Tensor, B: Tensor):
    r"""Returns the elements of A without the elements of B (Optimized)"""  # from: https://discuss.pytorch.org/t/any-way-of-filtering-given-rows-from-a-tensor-a/83828/2
    cdist = t.cdist(A.float(), B.float())
    min_dist = t.min(cdist, dim=1).values
    return A[min_dist > 0]


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
