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


class HyConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, weight_type=None, group_num=None) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_type = weight_type
        self.group_num = group_num
        self.theta = Parameter(torch.Tensor(in_dim, out_dim))
        if bias:
            self.bias = Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)
        if self.weight_type == 'group':
            self.group_weight = Parameter(torch.ones(self.group_num))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def MA_v(self, X, H):
        # message vertex ft
        X = X[H.node_idx]
        # aggregate vertex ft
        Y = scatter_add(X, H.hyedge_idx, dim=0)
        return Y

    def G_e(self, Y, H, group_weight=None):
        # generate hyperedge ft
        if self.weight_type == 'group':
            group_weight = group_weight if group_weight is not None else self.group_weight
            # weight = F.leaky_relu(self.group_weight)
            # weight = F.softmax(group_weight, dim=-1)
            weight = torch.sigmoid(group_weight)
            # weight = self.group_weight
            Y = H.add_group_weight(Y, weight)
        return Y

    def MA_e(self, X, Y, H):
        # message hyperedge ft
        Y = Y[H.hyedge_idx]
        # aggregate hyperedge ft
        X_new = scatter_add(Y, H.node_idx, dim=0)
        return X_new

    def U_v(self, X, X_new):
        # update vertex ft
        # X_new = X_new.matmul(self.theta)
        if self.bias is not None:
            X_new += self.bias
        return X_new

    def propagate(self, X, H, group_weight=None):

        X = X.matmul(self.theta)

        Y = self.MA_v(X, H)
        Y = self.G_e(Y, H, group_weight)

        X_new = self.MA_e(X, Y, H)
        X_new = self.U_v(X, X_new)

        return X_new

    def forward(self, X, H, group_weight=None):
        return self.propagate(X, H, group_weight=group_weight)
