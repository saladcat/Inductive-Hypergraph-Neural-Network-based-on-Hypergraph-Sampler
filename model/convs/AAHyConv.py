import torch
from torch_scatter import scatter_add
from .HyConv import HyConv


class AAHyConv(HyConv):

    def __init__(self, in_dim, out_dim, bias=True, weight_type=None, group_num=None) -> None:
        super().__init__(in_dim, out_dim, bias, weight_type, group_num)

        self.reset_parameters()

    def MA_v(self, X, H):
        # message vertex ft
        hyedge_norm = 1.0 / H.hyedge_degree().float()
        hyedge_norm[torch.isinf(hyedge_norm)] = 0
        X = X[H.node_idx] * hyedge_norm[H.hyedge_idx].unsqueeze(1)
        # aggregate vertex ft
        Y = scatter_add(X, H.hyedge_idx, dim=0)
        return Y

    def MA_e(self, X, Y, H):
        # message hyperedge ft
        node_norm = 1.0 / H.node_degree().float()
        node_norm[torch.isinf(node_norm)] = 0
        Y = Y[H.hyedge_idx] * node_norm[H.node_idx].unsqueeze(1)
        # aggregate hyperedge ft
        X_new = scatter_add(Y, H.node_idx, dim=0)
        return X_new
