import torch
from torch_scatter import scatter_max, scatter_add


def graph_count_node_num(A_idx, node_num=None):
    return node_num if node_num is not None else A_idx.flatten().max().item() + 1


def graph_node_degree(A_idx, node_num=None):
    if A_idx.dim() == 1:
        row = A_idx
    else:
        row, _ = A_idx
    node_num = graph_count_node_num(A_idx, node_num)
    src = torch.ones_like(row).float().to(A_idx.device)
    out = torch.zeros(node_num).to(A_idx.device)
    return out.scatter_add(0, row, src).long()


def graph_remove_self_loop(A_idx):
    row, col = A_idx
    mask = row != col
    mask = mask.unsqueeze(0).expand_as(A_idx)
    A_idx = A_idx[mask].view(2, -1)
    return A_idx


def graph_add_self_loop(A_idx, node_num=None):
    num_nodes = graph_count_node_num(A_idx, node_num)

    dtype, device = A_idx.dtype, A_idx.device
    loop = torch.arange(0, num_nodes, dtype=dtype, device=device)
    loop = loop.unsqueeze(0).repeat(2, 1)
    A_idx = torch.cat([A_idx, loop], dim=1)
    return A_idx


def graph_softmax(src, index, node_num=None):
    node_num = graph_count_node_num(index, node_num)

    out = src - scatter_max(src, index, dim=0, dim_size=node_num)[0][index]
    out = out.exp()
    out = out / (
        scatter_add(out, index, dim=0, dim_size=node_num)[index] +1e-16
    )
    return out
