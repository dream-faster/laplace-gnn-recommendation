from torch import Tensor
import torch as t


def intersection(t1: Tensor, t2: Tensor) -> Tensor:
    # This can be quite costly, seek for alternative implementations before using it in prod
    combined = t.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    intersection = uniques[counts > 1]
    return intersection


def difference(A: Tensor, B: Tensor):
    r"""Returns the elements of A without the elements of B (Optimized)"""
    cdist = t.cdist(A.float(), B.float())
    min_dist = t.min(cdist, dim=1).values
    return A[min_dist>0]
