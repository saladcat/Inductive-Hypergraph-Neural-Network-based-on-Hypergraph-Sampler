import torch
from torch_geometric.utils import add_remaining_self_loops
from torch_sparse.tensor import SparseTensor
import numpy as np
import torch_sparse
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import os
import os.path as osp
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from tqdm import tqdm

from data import HyNeighborSampler
from model import HyConvInd_ft

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device('cuda:3')

root = osp.join('/home/zhushiyang/code_repo/pyg_learning/data')
dataset = PygNodePropPredDataset('ogbn-products', root)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
data = dataset[0]
print(-1)
feature_n = dataset.num_features
class_n = dataset.num_classes

model = HyConvInd_ft(feature_n, [128, ], class_n).to(device)
x = data.x.to(device)
y = data.y.squeeze().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
edge_index = data.edge_index
# edge_index_loop = add_remaining_self_loops(edge_index)[0]

train_loader = HyNeighborSampler(edge_index, node_idx=split_idx['train'],
                                 sizes=[10, 5], batch_size=1024, shuffle=True,
                                 num_workers=12)
test_loader = HyNeighborSampler(edge_index, node_idx=None,
                                sizes=[10, 5], batch_size=1024, shuffle=False,
                                num_workers=12)

# train
epoch = 20

for i in range(epoch):
    model.train()

    pbar = tqdm(total=int(split_idx['train'].size(0)))
    pbar.set_description(f'Epoch {i:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        if batch_size != out.size(0) or batch_size != adjs[1].size[0][0]:
            print(batch_size, adjs[1].size[0][0], out.size(0))

        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)
    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(split_idx['train'].size(0))

    print("train:", loss, approx_acc)
#
xs = []
total_loss = total_correct = 0
pbar = tqdm(total=int(data.num_nodes))
pbar.set_description('Evaluating')

with torch.no_grad():
    for batch_size, n_id, adjs in test_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        if adjs[1].size[0][0] != batch_size:
            print(1)
        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        if batch_size != out.size(0):
            print(batch_size, adjs[1].size[0][0], out.size(0))

        xs.append(out)

        pbar.update(batch_size)
    pbar.close()
x_all = torch.cat(xs, dim=0)

y_true = y.cpu().unsqueeze(-1)
y_pred = x_all.argmax(dim=-1, keepdim=True)

train_acc = evaluator.eval({
    'y_true': y_true[split_idx['train']],
    'y_pred': y_pred[split_idx['train']],
})['acc']
val_acc = evaluator.eval({
    'y_true': y_true[split_idx['valid']],
    'y_pred': y_pred[split_idx['valid']],
})['acc']
test_acc = evaluator.eval({
    'y_true': y_true[split_idx['test']],
    'y_pred': y_pred[split_idx['test']],
})['acc']

print(train_acc, val_acc, test_acc)
