from .convs import HyConvInd
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter


class HyConvInd_ft(nn.Module):
    def __init__(self, ft_dim, hiddens, class_num, dropout=0.5, weight_type=None, group_num=None) -> None:
        super().__init__()
        self.dropout = dropout
        self.weight_type = weight_type
        self.group_num = group_num
        self.convs = []
        _in = ft_dim
        for h in hiddens:
            self.convs.append(HyConvInd(_in, h, weight_type=weight_type, group_num=group_num))
            _in = h
        self.convs.append(HyConvInd(hiddens[-1], class_num, weight_type=weight_type, group_num=group_num))
        self.convs = nn.ModuleList(self.convs)

        if self.weight_type == 'group':
            self.group_weight = Parameter(torch.ones(self.group_num))
        else:
            self.group_weight = None

    def forward(self, x, H, hyedge_weight=None):
        for i, (adj, _, size) in enumerate(H):
            # size(target_node,hypedge,node_n)
            s0, s1 = size

            x_target = x[:s0[0]]  # Target nodes are always placed first.
            # x = self.convs[i]((x, x_target), adj)
            x = self.convs[i](x, x_target, adj)
            if i != 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1)
