import numpy  as np
import torch
from scipy.sparse import coo_matrix
from SuperMoon.hygraph import hyedge_concat
from SuperMoon.hyedge import neighbor_distance

from utils import graph_add_self_loop, graph_remove_self_loop


def graph_count_node_num(A_idx, node_num=None):
    return node_num if node_num is not None else A_idx.flatten().max().item() + 1


def get_nerb(adj, n_hops=1):
    hop_adj = power_adj = np.array(adj)

    for i in range(n_hops - 1):
        power_adj = power_adj @ adj
        prev_hop_adj = hop_adj
        hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0)
    return hop_adj


def get_sparse_nerb(edge_idx, node_idx, n_hops=1):
    assert n_hops >= 1
    node_num = graph_count_node_num(edge_idx, None)

    row_idx, col_idx = edge_idx
    edge_idx = coo_matrix((np.ones_like(row_idx), (row_idx, col_idx)), shape=(node_num, node_num))
    H_idx = edge_idx.copy()
    while n_hops > 1:
        H_idx *= edge_idx
        n_hops -= 1
    H_idx = coo_matrix(H_idx)

    return torch.tensor(np.vstack((H_idx.row, H_idx.col)), dtype=torch.long)


def setup_graph(edge_idx, self_loop, approach_list, node_num, X_list=None):
    A_idx = None
    if approach_list is not None:
        for modality_idx, approaches in enumerate(approach_list):
            for approach in approaches:
                if approach.lower().startswith('knn'):
                    tmp_k = int(approach[approach.rfind('_') + 1:])
                    tmp_A_idx = neighbor_distance(torch.tensor(X_list[modality_idx]), tmp_k)[[1, 0]]
                else:
                    raise NotImplementedError
                A_idx = tmp_A_idx if A_idx is None else torch.cat([A_idx, tmp_A_idx], dim=-1)
    if edge_idx is not None:
        A_idx = torch.cat([edge_idx, A_idx], dim=-1)
    if self_loop:
        A_idx = graph_remove_self_loop(A_idx)
        A_idx = graph_add_self_loop(A_idx, node_num)
    else:
        A_idx = graph_remove_self_loop(A_idx)
    return A_idx


if __name__ == '__main__':
    adj = np.array([[1, 2], [1, 2]])
