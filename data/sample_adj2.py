from random import random, randrange
from typing import Optional, Tuple
import torch
from torch_sparse.tensor import SparseTensor
import torch_sparse
import numpy as np
from datetime import datetime
import hysample_cpp


def sample_adj_neib(src: SparseTensor, csc, csr, nodeset: torch.Tensor, hyedgeset: torch.Tensor,
                    num_neighbors: Tuple[int, int], replace: bool = False):
    rowptr, col, value = csr
    colptr, row, value = csc
    rowcount = src.storage.rowcount()
    colcount = src.storage.colcount()
    # log("sample hyedge")
    rowptr, col, new_hyedgeset = hysample_cpp.hysample_adj(rowptr, col, rowcount, nodeset, hyedgeset, num_neighbors[0])
    # log("sample node")
    colptr, row, new_nodeset = hysample_cpp.hysample_adj(colptr, row, colcount, new_hyedgeset, nodeset,
                                                         num_neighbors[1])

    # log("sample finsh")
    out_edge2node = SparseTensor(rowptr=rowptr, row=None, col=col,
                                 sparse_sizes=(nodeset.size(0), new_hyedgeset.size(0)),
                                 is_sorted=True)
    out_node2edge_t = SparseTensor(rowptr=colptr, row=None, col=row,
                                   sparse_sizes=(new_hyedgeset.size(0), new_nodeset.size(0)),
                                   is_sorted=True)
    out_node2edge = out_node2edge_t.t()

    out = (out_edge2node, out_node2edge)
    return out, new_nodeset, new_hyedgeset


def sample_(rowptr, col, row_ids: torch.Tensor, col_ids: torch.Tensor,
            num_neighbor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    col_id_list = col_ids.tolist()  # 用到的col id
    new_rowptr = torch.zeros(row_ids.size(0) + 1, dtype=torch.long)
    new_col_idxs = []  # col 新序列 col idx
    col_id_idx_dict = {col_id: i for i, col_id in enumerate(col_id_list)}
    cnt = 0
    for row_idx, row_id in enumerate(row_ids):  # 从row id遍历
        idx_n, idx_end = rowptr[row_id], rowptr[row_id + 1]
        idx_end_li = idx_n + num_neighbor
        idx_end = min(idx_end, idx_end_li)
        neighs = col[idx_n: idx_end]
        for idx, col_id in enumerate(neighs):
            cnt = cnt + 1
            if col_id not in col_id_idx_dict:  # 邻居 id 从来没用过
                col_id_idx_dict[col_id] = len(col_id_list)  # 加入id 并且idx当前最大的递增
                col_id_list.append(col_id)  # 放入已用到的col id
            new_col_idxs.append(col_id_idx_dict[col_id])  # 无论是否新的节点，col idx都要加该邻居为新的idx

        offset = len(new_col_idxs)  # offset_i-offset_{i-1}表示这个row_id已找到的邻居数量(有重复)，CSR用offset表示
        new_rowptr[row_idx + 1] = offset

    log(f"cnt:{cnt}")
    # convert list to tensor
    new_rowptr, new_col_idxs, col_id_list = new_rowptr, \
                                            torch.tensor(new_col_idxs), \
                                            torch.tensor(col_id_list)
    # 返回的是新的组织形式， 比如1和11，21，31相邻
    # 会返回new_rowptr,new_col_idxs,[1,11,21,31] 这样
    # col_idxs 表示原图中col_idxs
    # 所以要通过col_id_list再重新排序一下X ft matrix
    return new_rowptr, new_col_idxs, col_id_list


def log(string):
    print(f"{datetime.now()}:{string}")


if __name__ == '__main__':
    a = np.array([[1, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 1, 1],
                  [0, 0, 0, 1]])
    a = torch.tensor(a)
    b = SparseTensor.from_dense(a)

    print(1)
