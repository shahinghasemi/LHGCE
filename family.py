import torch
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

edge_index = torch.tensor([
    [0, 0, 1, 1, 2, 2, 3, 4, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11],
    [1, 2, 0, 2, 0, 1, 4, 3, 7, 8, 9, 10, 11, 6, 8, 9, 10, 11, 6, 7, 9, 10, 11, 6, 7, 8, 10, 11, 6, 7, 8, 9, 11, 6, 7, 8, 9, 10]
], dtype=torch.long)
# nNegative = (12 * 12) - (edge_index.shape[1]) - 12
# negative_index = negative_sampling(edge_index=edge_index, num_nodes=12, num_neg_samples=nNegative)
# train_edge = edge_index[:][:5]
x = torch.tensor([
    [1, 1],
    [2, 0],
    [1, 1],
    [0, 1],
    [1, 0],
    [0, 0],
    [3, 2],
    [3, 2],
    [3, 2],
    [4, 1],
    [4, 1],
    [3, 2],
], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print('org.edge_index: ', data.edge_index)
print('------------------------')
train, val, test = RandomLinkSplit(0.0, 0.8, is_undirected=False)(data)
print('train: ', train)
print('------------------------')
print('val: ', val)
print('------------------------')
print('test: ', test)
