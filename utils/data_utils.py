import numpy as np
import scipy.sparse as sp
import torch


def idx_encode(labels):
    classes = list(set(labels))  # 这里可能出现每次不一样 所以sort
    classes.sort()
    dict_label2idx = {item: i for i, item in enumerate(classes)}
    ret = np.array(list(map(dict_label2idx.get, labels)))
    return ret


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def ft_norm(X):
    rowsum = X.sum(dim=1, keepdim=True)
    r_inv = rowsum.pow(-1)
    r_inv[torch.isinf(r_inv)] = 0.
    X_normed = X * r_inv
    return X_normed
