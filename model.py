from metrics import calculateMetric
from torch_geometric.nn import SAGEConv, to_hetero, GATv2Conv
import torch
from torch.nn import Linear, ModuleList

class GNNEncoder(torch.nn.Module):
    def __init__(self, neurons, layers):
        super().__init__()
        self.layers = layers
        self.convs = ModuleList([SAGEConv((-1, -1), neurons) for i in range(self.layers)])

    def forward(self, x, edge_index):
        for i, l in enumerate(self.convs):
            if i == self.layers - 1:
                x = self.convs[i](x, edge_index)
            else:
                x = self.convs[i](x, edge_index).relu()

        return x

class GNNEncoder2(torch.nn.Module):
    def __init__(self, neurons):
        super().__init__()
        self.conv = GATv2Conv((-1, -1), neurons, add_self_loops=False)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index).relu()

        return x

class EdgeDecoder(torch.nn.Module):
    def __init__(self, neurons, ):
        super().__init__()
        self.lin1 = Linear(2 * neurons, neurons)
        self.lin2 = Linear(neurons, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['drug'][row], z_dict['disease'][col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class Model(torch.nn.Module):
    def __init__(self, data, neurons, layers, encoderType):
        super().__init__()
        if encoderType == 'SAGE':
            self.encoder = GNNEncoder(neurons, layers)
        elif encoderType == 'attention':
            self.encoder = GNNEncoder2(neurons)

        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(neurons)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


def train(data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    pred = model(data.x_dict, data.edge_index_dict, data['drug', 'treats', 'disease'].edge_label_index)
    loss = criterion(pred, data['drug', 'treats', 'disease'].edge_label)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(test_data, model, thresholdPercent):
    model.eval()
    pred = model(test_data.x_dict, test_data.edge_index_dict, test_data['drug','treats', 'disease'].edge_label_index)

    pred = pred.detach().numpy()
    edge_label = test_data['drug','treats', 'disease'].edge_label.detach().numpy()

    edge_label_index = test_data['drug', 'treats', 'disease'].edge_label_index
    metrics = calculateMetric(edge_label, pred, edge_label_index, edge_label, thresholdPercent)
    return metrics

