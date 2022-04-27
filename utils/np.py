import numpy as np


def np_groupby_first_col(a: np.ndarray) -> np.ndarray:
    """
    Group an array by the first axis of the array.
    """
    return np.array(np.split(a[:, 1], np.unique(a[:, 0], return_index=True)[1][1:]))
