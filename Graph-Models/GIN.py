import torch
import torch.nn as nn
import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn.functional as F

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


class GINNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GINNet, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        dim1 = 32
        nn1 = nn.Sequential(nn.Linear(self.num_features, dim1), nn.ReLU(), nn.Linear(dim1, dim1))
        self.conv1 = GINConv(nn1)
        self.bn1 = nn.BatchNorm1d(dim1)

        dim2 = 32
        nn2 = nn.Sequential(nn.Linear(dim1, dim2), nn.ReLU(), nn.Linear(dim2, self.num_classes))
        self.conv2 = GINConv(nn2)


    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.conv2(x, edge_index)
        return x

device = torch.device('cpu')
model = GINNet(dataset.num_features, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.batch)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

@torch.no_grad()
def test(data):
    model.eval()
    out, accs = model(data.x, data.edge_index, data.batch), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

for epoch in range(1, 201):
    train(data)
    train_acc, val_acc, test_acc = test(data)
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')