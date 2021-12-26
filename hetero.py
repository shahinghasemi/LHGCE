import torch_geometric.transforms as T
import prepareData
from torch_geometric.nn import SAGEConv, to_hetero
import torch
from torch.nn import Linear
import torch.nn.functional as F
from metrics import calculateMetric
import torch_geometric
import numpy as np

torch_geometric.seed_everything(3)
torch.manual_seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

DRUG_NUMBER = 269
DISEASE_NUMBER = 598
ENZYME_NUMBER = 108
STRUCTURE_NUMBER = 881
PATHWAY_NUMBER = 258
TARGET_NUMBER = 529
INTERACTIONS_NUMBER = 18416
NONINTERACTIONS_NUMBER = 142446
FOLDS = 5
EPOCHS = 500

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['drug'][row], z_dict['disease'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class Model(torch.nn.Module):
    def __init__(self, data, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

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
def test(test_data, model):
    model.eval()
    pred = model(test_data.x_dict, test_data.edge_index_dict, test_data['drug','treats', 'disease'].edge_label_index)

    pred = pred.detach().numpy()
    edge_label = test_data['drug','treats', 'disease'].edge_label.detach().numpy()

    metrics = calculateMetric(edge_label, pred, 3)
    return metrics


def main():
    drugDisease = np.loadtxt('./data/drug_disease.csv', delimiter=',')
    totalInteractions = np.array(np.mat(np.where(drugDisease == 1)).T) #(18416, 2)
    totalNonInteractions = np.array(np.mat(np.where(drugDisease == 0)).T) #(142446, 2)

    selectedInteractions, selectedNonInteractions = prepareData.splitter(100, 100, totalInteractions, totalNonInteractions)

    interactionsIndicesFolds, nonInteractionsIndicesFolds = prepareData.foldify(selectedInteractions, selectedNonInteractions)

    data = prepareData.createHeteroNetwork(['pathway', 'target', 'structure', 'enzyme'])

    metrics = np.zeros(7)

    for k in range(FOLDS):
        messageEdgesIndex, superVisionEdgesIndex, testEdgesIndex = prepareData.splitEdgesBasedOnFolds(interactionsIndicesFolds, k)

        edge_index = [[], []]
        for drugIndex, diseaseIndex in selectedInteractions[messageEdgesIndex]:
            edge_index[0].append(drugIndex)
            edge_index[1].append(diseaseIndex)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        data['drug', 'treats', 'disease'].edge_index = edge_index

        # Training
        edge_label_index = [[], []]
        neg_edge_index = [[], []]
        edge_label = []
        for drugIndex, diseaseIndex in selectedInteractions[superVisionEdgesIndex]:
            edge_label_index[0].append(drugIndex)
            edge_label_index[1].append(diseaseIndex)
            edge_label.append(1)
        for drugIndex, diseaseIndex in selectedNonInteractions:
            neg_edge_index[0].append(drugIndex)
            neg_edge_index[1].append(diseaseIndex)
            edge_label.append(0)
        edge_label_index = torch.tensor(edge_label_index, dtype=torch.long)
        neg_edge_index = torch.tensor(neg_edge_index, dtype=torch.long)
        edge_label = torch.tensor(edge_label, dtype=torch.float)

        data['drug', 'treats', 'disease'].edge_label_index = torch.cat([edge_label_index, neg_edge_index],dim=-1)
        data['drug', 'treats', 'disease'].edge_label = edge_label

        # TODO: maybe undirected reduce the performance
        data = T.ToUndirected()(data)
        data = T.AddSelfLoops()(data)
        data = T.NormalizeFeatures()(data)

        model = Model(data, hidden_channels=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Due to lazy initialization, we need to run one model step so the number
        # of parameters can be inferred:
        with torch.no_grad():
            model.encoder(data.x_dict, data.edge_index_dict)

        criterion = torch.nn.BCEWithLogitsLoss()
        for epoch in range(1, EPOCHS):
            loss = train(data, model, optimizer, criterion)
            if epoch % 10 == 0:
                print('epoch: ', epoch, 'train loss: ', loss)
        
        # Testing
        edge_label_index = [[], []]
        neg_edge_index = [[], []]
        edge_label = []
        for drugIndex, diseaseIndex in selectedInteractions[testEdgesIndex]:
            edge_label_index[0].append(drugIndex)
            edge_label_index[1].append(diseaseIndex)
            edge_label.append(1)
        for drugIndex, diseaseIndex in selectedNonInteractions:
            neg_edge_index[0].append(drugIndex)
            neg_edge_index[1].append(diseaseIndex)
            edge_label.append(0)
        edge_label_index = torch.tensor(edge_label_index, dtype=torch.long)
        neg_edge_index = torch.tensor(neg_edge_index, dtype=torch.long)
        edge_label = torch.tensor(edge_label, dtype=torch.float)
        
        data['drug', 'treats', 'disease'].edge_label_index = torch.cat([edge_label_index, neg_edge_index], dim=-1)
        data['drug', 'treats', 'disease'].edge_label = edge_label
        
        metric = test(data, model)
        metrics += metric

        print('metric: ', metric)
    return metrics

metrics = main()
print('results: ', metrics / FOLDS)
