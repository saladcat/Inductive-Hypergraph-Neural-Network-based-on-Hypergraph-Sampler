import copy
from typing import List, Optional, Tuple, NamedTuple, Union

import torch
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
import torch_sparse
import datetime
from datetime import datetime
from data import sample_adj, sample_adj_neib


class Adj(NamedTuple):
    edge_index: Tuple[SparseTensor, SparseTensor]
    e_id: int
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(tuple(i.to(*args, **kwargs) for i in self.edge_index),
                   0,
                   self.size)


class HyNeighborSampler(torch.utils.data.DataLoader):

    def __init__(self, edge_index: torch.Tensor, sizes: List[int],
                 node_idx: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 hyedge_num=None, **kwargs):

        N = int(edge_index[0].max() + 1) if num_nodes is None else num_nodes
        M = int(edge_index[1].max() + 1) if num_nodes is None else hyedge_num

        edge_attr = torch.arange(edge_index.size(1))
        adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                           value=edge_attr, sparse_sizes=(N, M),
                           is_sorted=False)
        self.adj = adj.to('cpu')
        self.csc = adj.csc()
        self.csr = adj.csr()

        if node_idx is None:
            node_idx = torch.arange(N)
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        self.sizes = sizes

        super(HyNeighborSampler, self).__init__(node_idx.tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)
        adjs: List[Adj] = []

        node_ids = batch
        hyedge_ids = torch.empty(0).long()

        for size in self.sizes:
            out, node_ids, hyedge_ids = sample_adj_neib(self.adj, self.csc, self.csr, node_ids, hyedge_ids,
                                                        (size, size), False)

            size = [adj.sparse_sizes() for adj in out]
            adjs.append(Adj(out, 0, size))
            # adjs.append(Adj(edge_index, e_id, size))

        if len(adjs) > 1:
            return batch_size, node_ids, adjs[::-1]
        else:
            return batch_size, node_ids, adjs[0]

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)


def log(string):
    print(f"{datetime.now()}:{string}")
