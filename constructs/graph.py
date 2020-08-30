from scipy.spatial.distance import cdist
import numpy as np
from .helper import sigma, Eu_dis


def construct_graph(X, self_loop=True, kth=8):
    x_dist = cdist(X, X)
    # A = np.exp(-(x_dist / sigma(x_dist)))
    #
    # # Convert to symmetric matrix
    # A = 0.5 * (A + A.T)

    # filter other edges
    knns = np.argpartition(x_dist, kth, axis=-1)[:, :kth]
    # knns = np.argsort(x_dist, axis=-1)[:, :kth]

    col = np.arange(knns.shape[0]).repeat(knns.shape[1])
    row = knns.reshape(-1)
    A = np.zeros_like(x_dist)
    A[row, col] = 1.0

    if self_loop:
        A[np.diag_indices_from(A)] = 1.0
    else:
        A[np.diag_indices_from(A)] = 0

    # norm
    rowsum = np.array(A.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    A = np.array(A.dot(d_mat_inv_sqrt.transpose()).dot(d_mat_inv_sqrt))
    return A

