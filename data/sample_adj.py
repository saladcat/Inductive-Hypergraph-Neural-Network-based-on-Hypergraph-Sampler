from typing import Optional, Tuple
import torch
from torch_sparse.tensor import SparseTensor
import torch_sparse
import numpy as np


def sample_adj(src: SparseTensor, nodeset: torch.Tensor, hyedgeset: torch.Tensor,
               num_neighbors: Tuple[int, int], replace: bool = False):
    rowptr, col, value = src.csr()
    colptr, row, value = src.csc()

    rowcount = src.storage.rowcount()
    colcount = src.storage.colcount()

    rowptr, col, e_id_edge, new_hyedgeset = sample_(rowptr, col, rowcount, nodeset, hyedgeset,
                                                    num_neighbors[0], replace)

    out_edge2node = SparseTensor(rowptr=rowptr, row=None, col=col, value=e_id_edge,
                                 sparse_sizes=(nodeset.size(0), new_hyedgeset.size(0)),
                                 is_sorted=True)

    colptr, row, e_id_node, new_nodeset = sample_(colptr, row, colcount, new_hyedgeset, nodeset,
                                                  num_neighbors[1], replace)

    out_node2edge_t = SparseTensor(rowptr=colptr, row=None, col=row, value=e_id_node,
                                   sparse_sizes=(new_hyedgeset.size(0), new_nodeset.size(0)),
                                   is_sorted=True)
    out_node2edge = out_node2edge_t.t()

    out = (out_edge2node, out_node2edge)
    return out, new_nodeset, new_hyedgeset


def sample_(rowptr, col, rowcount, row_ids: torch.Tensor, col_ids: torch.Tensor,
            num_neighbor, replace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    col_id_list = col_ids.tolist()  # 用到的col id
    new_rowptr = torch.zeros(row_ids.size(0) + 1, dtype=torch.long)
    new_col_idxs = []  # col 新序列 col idx
    col_id_idx_dict = {}  # col id->idx
    col_idxs = []
    for i, col_id in enumerate(col_id_list):
        col_id_idx_dict[col_id] = i  # col id 无论多大 idx从0开始

    for row_idx, row_id in enumerate(row_ids):  # 从row id遍历
        for i in range(rowcount[row_id]):  # row id 对应邻居遍历
            col_idx = rowptr[row_id] + i  # 邻居 idx
            col_id = col[col_idx].item()  # 邻居 id

            if col_id_idx_dict.get(col_id) is None:  # 邻居 id 从来没用过
                col_id_idx_dict[col_id] = len(col_id_list)  # 加入id 并且idx当前最大的递增
                col_id_list.append(col_id)  # 放入已用到的col id
            col_idxs.append(col_idx)
            new_col_idxs.append(col_id_idx_dict[col_id])  # 无论是否新的节点，col idx都要加该邻居为新的idx

        offset = len(new_col_idxs)  # offset_i-offset_{i-1}表示这个row_id已找到的邻居数量(有重复)，CSR用offset表示
        new_rowptr[row_idx + 1] = offset

    # convert list to tensor
    new_rowptr, new_col_idxs, col_id_list, col_idxs = new_rowptr, \
                                                      torch.tensor(new_col_idxs), \
                                                      torch.tensor(col_id_list), \
                                                      torch.tensor(col_idxs)
    # 返回的是新的组织形式， 比如1和11，21，31相邻
    # 会返回new_rowptr,new_col_idxs,col_idxs,[1,11,21,31] 这样
    # col_idxs 表示原图中col_idxs
    # 所以要通过col_id_list再重新排序一下X ft matrix
    return new_rowptr, new_col_idxs, col_idxs, col_id_list


if __name__ == '__main__':
    a = np.array([[1, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 1, 1],
                  [0, 0, 0, 1]])
    a = torch.tensor(a)
    b = SparseTensor.from_dense(a)

    c = sample_adj(b, torch.tensor([0]), torch.tensor([0]), (0, 0), False)

    print(1)
