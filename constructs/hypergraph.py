from copy import deepcopy

import torch
from SuperMoon.hygraph import hyedge_concat
from SuperMoon.hyedge import neighbor_distance, self_loop_add, count_node
import numpy as np
from scipy.sparse import coo_matrix

from utils import get_node_sparse_subgraph
from utils.graph_utils import graph_count_node_num


def hg_edge(edge_idx):
    num_edge = edge_idx.size(1)
    hyedge_idx = torch.arange(num_edge, dtype=torch.long).view(1, -1)
    hyedge_idx = hyedge_idx.repeat((2, 1))
    H_idx = torch.cat([edge_idx.contiguous().view(1, -1), hyedge_idx.contiguous().view(1, -1)], dim=0)
    return H_idx


def hg_hop_j(edge_idx, j, node_num=None):
    assert j >= 1
    node_num = graph_count_node_num(edge_idx, node_num)

    edge_idx = edge_idx.numpy()
    row_idx, col_idx = edge_idx
    edge_idx = coo_matrix((np.ones_like(row_idx), (row_idx, col_idx)), shape=(node_num, node_num))
    H_idx = edge_idx.copy()
    while j > 1:
        H_idx *= edge_idx
        j -= 1
    H_idx = coo_matrix(H_idx)
    return torch.tensor(np.vstack((H_idx.row, H_idx.col)), dtype=torch.long)


def hg_raw(H_idx):
    return torch.tensor(H_idx, dtype=torch.long)


def hg_KNN_k(X, k):
    return neighbor_distance(X, k)


def hg_rand_n_k(n, k, node_n):
    '''
    :param n: # of hyedge
    :param k: # of node per hyedge
    :param node_n: # of node
    :return: H_idx [node_idx,hyedge_idx]
    '''
    H_idx = []
    for hyedge_idx in range(n):
        y = torch.full([1, k], hyedge_idx, dtype=torch.long)
        x = torch.randint(node_n, [1, k])  # sample same x possibly
        x[0, 0] = hyedge_idx
        temp_H_idx = torch.cat([x, y], dim=0)
        H_idx.append(temp_H_idx)

    H_idx = torch.cat(H_idx, dim=1)
    return H_idx


class Hypergraph:

    def __init__(self, X_list, approach_list, raw_edge_idx=None, node_num=None):
        self.device = 'cpu'
        self.X_list = X_list
        self.approach_list = approach_list
        self.raw_edge_idx = raw_edge_idx
        self.node_num = node_num

        self.groups = []
        self.group_hyedge_nums = []
        self.incidence_matrix_idx = None

        self._setup()
        self.node_num = self.count_node_num(node_num)
        self.hyedge_num = self.count_hyedge_num()

    def _setup(self):
        H_idx = []
        for modality_idx, approaches in enumerate(self.approach_list):
            for approach in approaches:
                approach: str
                if approach.lower().startswith('hop'):
                    tmp_j = int(approach[approach.rfind('_') + 1:])
                    tmp_H_idx = hg_hop_j(self.raw_edge_idx, tmp_j, self.node_num)
                elif approach.lower().startswith('edge'):
                    tmp_H_idx = hg_edge(self.raw_edge_idx)
                elif approach.lower().startswith('knn'):
                    tmp_k = int(approach[approach.rfind('_') + 1:])
                    tmp_H_idx = hg_KNN_k(self.X_list[modality_idx], tmp_k)
                elif approach.lower().startswith('rand'):
                    approach_split_list = approach.split('_')
                    assert len(approach_split_list) == 3
                    tmp_n = int(approach_split_list[1])
                    tmp_k = int(approach_split_list[2])
                    tnp_node_n = self.X_list[modality_idx].shape[0]
                    tmp_H_idx = hg_rand_n_k(tmp_n, tmp_k, tnp_node_n)
                elif approach.lower().startswith('raw'):
                    tmp_H_idx = hg_raw(self.raw_edge_idx)
                else:
                    raise NotImplementedError
                self.groups.append(tmp_H_idx)
                self.group_hyedge_nums.append(self.count_hyedge_num(hyedge_idx=tmp_H_idx))
                H_idx.append(tmp_H_idx)
        self.incidence_matrix_idx = hyedge_concat(H_idx)

    @property
    def idx(self):
        return self.incidence_matrix_idx[0], self.incidence_matrix_idx[1]

    @property
    def node_idx(self):
        return self.incidence_matrix_idx[0]

    @property
    def hyedge_idx(self):
        return self.incidence_matrix_idx[1]

    def to(self, device):
        self.device = device
        for _idx in range(len(self.groups)):
            self.groups[_idx] = self.groups[_idx].to(device)
        self.incidence_matrix_idx = self.incidence_matrix_idx.to(device)
        return self

    def count_node_num(self, node_num=None):
        return self.node_idx.max().item() + 1 if node_num is None else node_num

    def count_hyedge_num(self, hyedge_num=None, hyedge_idx=None):
        hyedge_idx = self.hyedge_idx if hyedge_idx is None else hyedge_idx
        return hyedge_idx.max().item() + 1 if hyedge_num is None else hyedge_num

    def node_degree(self):
        src = torch.ones_like(self.node_idx).float().to(self.device)
        out = torch.zeros(self.node_num).to(self.device)
        return out.scatter_add(0, self.node_idx, src).float()

    def hyedge_degree(self):
        src = torch.ones_like(self.hyedge_idx).float().to(self.device)
        out = torch.zeros(self.hyedge_num).to(self.device)
        return out.scatter_add(0, self.hyedge_idx, src).float()

    def group_num(self):
        return len(self.group_hyedge_nums)

    def add_group_weight(self, hyedge_ft, group_weights):
        device = group_weights.device
        weights = []
        for _idx, num in enumerate(self.group_hyedge_nums):
            weights.append(torch.ones(num, device=device) * group_weights[_idx])
        weights = torch.cat(weights)
        return hyedge_ft * weights.unsqueeze(1)

    def get_neibs(self, node_idx, n_hops=2):
        assert node_idx < self.node_num
        node_idxs, hyedge_idxs = get_node_sparse_subgraph(node_idx, self.incidence_matrix_idx, n_hops=n_hops)
        return node_idxs, hyedge_idxs

    def get_subgraph(self, node_idx, n_hops):
        # error
        node_idxs, hyedge_idxs = self.get_neibs(node_idx, n_hops)
        node_list = self.node_idx
        node_idxs = node_idxs.unsqueeze(1)
        node_list_repeat = node_list.repeat((node_idxs.size(0), 1))
        node_list_flag = (node_list_repeat == node_idxs).any(0)

        hyedge_list = self.hyedge_idx
        hyedge_idxs = hyedge_idxs.unsqueeze(1)
        hyedge_list_repeat = hyedge_list.repeat((hyedge_idxs.size(0), 1))
        hyedge_list_flag = (hyedge_list_repeat == hyedge_idxs).any(0)

        H_mask = torch.stack([node_list_flag, hyedge_list_flag], dim=0).all(0)
        use_node = torch.masked_select(node_list, H_mask)
        use_hyedge = torch.masked_select(hyedge_list, H_mask)

        incidence_matrix_idx = torch.stack([use_node, use_hyedge], dim=0)

        subgraph = deepcopy(self)
        subgraph.incidence_matrix_idx = incidence_matrix_idx
        return subgraph


if __name__ == '__main__':
    a = torch.tensor([1, 2, 3, 1, 2, 3, 4, 3])
    b = a.clone()

    c = torch.stack([a, b], dim=0)
    mask = torch.tensor([0, 1, 1, 0, 0, 0, 0, 1], dtype=torch.bool).repeat((2, 1))
    print(c)
    print(c.masked_select(mask))
