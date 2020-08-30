from .convs import AAHyConv
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter


class AAHyConv_ft(nn.Module):
    def __init__(self, ft_dim, hiddens, class_num, dropout=0.5, weight_type=None, group_num=None) -> None:
        super().__init__()
        self.dropout = dropout
        self.weight_type = weight_type
        self.group_num = group_num
        self.convs = []
        _in = ft_dim
        for h in hiddens:
            self.convs.append(AAHyConv(_in, h, weight_type=weight_type, group_num=group_num))
            _in = h
        self.convs = nn.ModuleList(self.convs)
        self.out_conv = AAHyConv(hiddens[-1], class_num, weight_type=weight_type, group_num=group_num)

        if self.weight_type == 'group':
            self.group_weight = Parameter(torch.ones(self.group_num))
        else:
            self.group_weight = None

    def forward(self, X, H, hyedge_weight=None):
        for h in self.convs:
            X = F.dropout(X, self.dropout, training=self.training)
            X = h(X, H, group_weight=self.group_weight)
            X = F.leaky_relu(X, inplace=True)

        X = F.dropout(X, self.dropout, training=self.training)
        X = self.out_conv(X, H, group_weight=self.group_weight)
        return X
