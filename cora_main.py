import torch
from torch_sparse.tensor import SparseTensor
import numpy as np
import torch_sparse
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import os
from tqdm import tqdm

from constructs import Hypergraph
from data import HyNeighborSampler
from model import HyConvInd_ft

device = torch.device('cuda:6')

dataset = Planetoid(root='data/Cora', name="Cora")

data = dataset[0]

feature_n = dataset.num_features
class_n = dataset.num_classes

model = HyConvInd_ft(feature_n, [128, ], class_n).to(device)

x = data.x.to(device)
y = data.y.squeeze().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

H = Hypergraph([x, ], [['raw'], ], raw_edge_idx=data.edge_index)
edge_index = data.edge_index
train_loader = HyNeighborSampler(edge_index, node_idx=data.train_mask,
                                 sizes=[-1, -1], batch_size=1024, shuffle=True,
                                 num_workers=12)

val_loader = HyNeighborSampler(edge_index, node_idx=data.val_mask,
                               sizes=[25, 10], batch_size=1024, shuffle=False,
                               num_workers=1)

test_loader = HyNeighborSampler(edge_index, node_idx=data.test_mask,
                                sizes=[25, 10], batch_size=1000, shuffle=False,
                                num_workers=1)


@torch.no_grad()
def valaa(graph_loader, is_val=True):
    model.eval()

    pbar = tqdm(total=int(data.val_mask.sum()))
    pbar.set_description(f'Epoch {i:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in graph_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        out = model(x[n_id], adjs)

        loss = F.nll_loss(out, y[n_id[:batch_size]])
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)
    pbar.close()

    loss = total_loss / len(graph_loader)
    approx_acc = total_correct / int(data.val_mask.sum())
    if not is_val:
        approx_acc = total_correct / int(data.test_mask.sum())
    print("val:", loss, approx_acc)


# train
epoch = 20

for i in range(epoch):
    model.train()

    pbar = tqdm(total=int(data.train_mask.sum()))
    pbar.set_description(f'Epoch {i:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)
    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    print("train:", loss, approx_acc)
    valaa(val_loader)
print('(---------------)')
valaa(test_loader, is_val=False)
print('(---------------)')
