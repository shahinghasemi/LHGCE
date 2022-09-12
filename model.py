from metrics import calculateMetric
from torch_geometric.nn import SAGEConv, to_hetero, GATv2Conv, GraphSAGE
import torch
from torch.nn import Linear, ModuleList, Sequential, ReLU

class GNNEncoder(torch.nn.Module):
    def __init__(self, neurons, layers):
        super().__init__()
        self.layers = layers
        self.convs = ModuleList([SAGEConv((-1, -1), neurons, normalize=True) for i in range(self.layers)])

    def forward(self, x, edge_index):
        for i, l in enumerate(self.convs):
            if i == self.layers - 1:
                x = self.convs[i](x, edge_index)
            else:
                x = self.convs[i](x, edge_index).relu()

        return x


class Linears(torch.nn.Module):
    def __init__(self, neurons, aggregator):
        super().__init__()

        if aggregator == 'concatenate':
            self.linear = Sequential(
                Linear(2 * neurons, neurons),
                ReLU(),
                Linear(neurons, 1)
            )
        elif aggregator == 'mean' or aggregator == 'sum' or aggregator == 'mul':
            self.linear = Sequential(
                Linear(neurons, neurons),
                ReLU(),
                Linear(neurons, 1)
            )       

        self.aggregator = aggregator

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        if self.aggregator == 'concatenate':
            z = torch.cat([z_dict['drug'][row], z_dict['disease'][col]], dim=-1)

        elif self.aggregator == 'mean':
            drug = z_dict['drug'][row]
            disease = z_dict['disease'][col]
            z = ((drug + disease) /2)

        elif self.aggregator == 'sum':
            drug = z_dict['drug'][row]
            disease = z_dict['disease'][col]
            z = drug + disease 

        elif self.aggregator == 'mul':
            drug = z_dict['drug'][row]
            disease = z_dict['disease'][col]
            z = drug * disease 

        z = self.linear(z)
        return z.view(-1)

class Model(torch.nn.Module):
    def __init__(self, data, neurons, layers, aggregator):
        super().__init__()
        self.encoder = GNNEncoder(neurons, layers)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.linear = Linears(neurons, aggregator)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.linear(z_dict, edge_label_index)


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

