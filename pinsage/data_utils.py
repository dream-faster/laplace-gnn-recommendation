import dgl
import numpy as np
import scipy.sparse as ssp


def build_val_test_matrix(g, val_indices, test_indices, utype, itype, etype):
    n_users = g.number_of_nodes(utype)
    n_items = g.number_of_nodes(itype)
    val_src, val_dst = g.find_edges(val_indices, etype=etype)
    test_src, test_dst = g.find_edges(test_indices, etype=etype)
    val_src = val_src.numpy()
    val_dst = val_dst.numpy()
    test_src = test_src.numpy()
    test_dst = test_dst.numpy()
    val_matrix = ssp.coo_matrix(
        (np.ones_like(val_src), (val_src, val_dst)), (n_users, n_items)
    )
    test_matrix = ssp.coo_matrix(
        (np.ones_like(test_src), (test_src, test_dst)), (n_users, n_items)
    )

    return val_matrix, test_matrix


def linear_normalize(values):
    return (values - values.min(0, keepdims=True)) / (
        values.max(0, keepdims=True) - values.min(0, keepdims=True)
    )
