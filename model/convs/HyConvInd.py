import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch_scatter import scatter_add

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch_scatter import scatter_add
from torch_sparse import SparseTensor


class HyConvInd(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, weight_type=None, group_num=None) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_type = weight_type
        self.group_num = group_num
        self.theta = Parameter(torch.Tensor(in_dim, out_dim))
        # self.self_theta = Parameter(torch.Tensor(in_dim, out_dim))
        if bias:
            self.bias = Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)
        if self.weight_type == 'group':
            self.group_weight = Parameter(torch.ones(self.group_num))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)
        # nn.init.xavier_uniform_(self.self_theta)

    def MA_v(self, X, H: SparseTensor):
        # message vertex ft
        hyedge_norm = 1.0 / H.storage.colcount().float()

        hyedge_norm[torch.isinf(hyedge_norm)] = 0
        X = X[H.storage.row()] * hyedge_norm[H.storage.col()].unsqueeze(1)
        # aggregate vertex ft
        Y = scatter_add(X, H.storage.col(), dim=0)
        return Y

    def G_e(self, Y, H: SparseTensor, group_weight=None):
        # generate hyperedge ft
        if self.weight_type == 'group':
            group_weight = group_weight if group_weight is not None else self.group_weight
            weight = torch.sigmoid(group_weight)
            Y = H.add_group_weight(Y, weight)
        return Y

    def MA_e(self, X, Y, H: SparseTensor):
        # message hyperedge ft
        node_norm = 1.0 / H.storage.rowcount().float()
        node_norm[torch.isinf(node_norm)] = 0
        Y = Y[H.storage.col()] * node_norm[H.storage.row()].unsqueeze(1)
        # aggregate hyperedge ft
        X_new = scatter_add(Y, H.storage.row(), dim=0, dim_size=H.size(0))
        return X_new

    def U_v(self, X, X_new, X_target):
        # update vertex ft
        if X_target is not None:
            # X_target = X_target.matmul(self.self_theta)
            X_new += X_target
        if self.bias is not None:
            X_new += self.bias
        return X_new

    def propagate(self, X, X_target, H, group_weight=None):
        e2n, n2e = H

        Y = self.MA_v(X, n2e)
        Y = self.G_e(Y, n2e, group_weight)

        X_new = self.MA_e(X, Y, e2n)
        X_new = self.U_v(X, X_new, X_target)

        return X_new

    def forward(self, X, X_target, H, group_weight=None):
        X = X.matmul(self.theta)
        return self.propagate(X, None, H, group_weight=group_weight)
