import torch_geometric.transforms as T
from prepareData import splitter, foldify, createHeteroNetwork, splitEdgesBasedOnFolds
import torch
import argparse
import torch_geometric
import numpy as np
from model import Model, test, train
from dataloader import dataloader

# Make everything reproducible
torch_geometric.seed_everything(3)
torch.manual_seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

# Parsing CLI args.
parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--dataset', help='dataset to use', type=str, default='lagcn')
parser.add_argument('--epochs', help='number of epochs to train in model',type=int, default=20)
parser.add_argument('--thr-percent', help='the threshold percentage with respect to batch size',type=int, default=5)
parser.add_argument('--dropout', help='dropout probability for DNN',type=float, default=0.3)
parser.add_argument('--lr', help='learning rate for DNN',type=float, default=0.01)
parser.add_argument('--agg', help='aggregation method for Linear layer to predict', type=str, default='concatenate')
parser.add_argument('--l', help='number of layers for graph convolutional encoder', type=int, default=2)
parser.add_argument('--n', help='number of neurons for each GCE layer', type=int, default=32)
parser.add_argument('--same', help='whether the same number of negatives should be selected as positives(interations)', type=lambda x: (str(x).lower() == 'true'), default=False)

args = parser.parse_args()
print(args)

# Setting the dynamic global variables
DATASET = args.dataset
EPOCHS = args.epochs
DROPOUT = args.dropout #useless
THRESHOLD_PERCENT = args.thr_percent
LEARNING_RATE= args.lr
AGGREGATOR = args.agg
LAYERS = args.l
NEURONS = args.n
SAME_NEGATIVE = args.same

# Setting the static global variables
# DRUG_NUMBER = 269
# DISEASE_NUMBER = 598
# ENZYME_NUMBER = 108
# STRUCTURE_NUMBER = 881
# PATHWAY_NUMBER = 258
# TARGET_NUMBER = 529
# INTERACTIONS_NUMBER = 18416
# NONINTERACTIONS_NUMBER = 142446
FOLDS = 5

def main():
    data, totalInteractions, totalNonInteractions = dataloader(DATASET)

    selectedInteractions, selectedNonInteractions = splitter(DATASET, SAME_NEGATIVE, totalInteractions, totalNonInteractions)
    interactionsIndicesFolds, nonInteractionsIndicesFolds = foldify(selectedInteractions, selectedNonInteractions)

    metrics = np.zeros(7)

    for k in range(FOLDS):
        messageEdgesIndex, superVisionEdgesIndex, testEdgesIndex = splitEdgesBasedOnFolds(interactionsIndicesFolds, k)

        testNonEdgesIndex = nonInteractionsIndicesFolds[k]
        trainNonEdgesIndex = np.setdiff1d(nonInteractionsIndicesFolds.flatten(), testNonEdgesIndex, assume_unique=True)

        edge_index = [[], []]
        for drugIndex, diseaseIndex in selectedInteractions[messageEdgesIndex]:
            edge_index[0].append(drugIndex)
            edge_index[1].append(diseaseIndex)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        data['drug', 'treats', 'disease'].edge_index = edge_index

        #------------------Training------------------#
        edge_label_index = [[], []]
        neg_edge_index = [[], []]
        edge_label = []
        for drugIndex, diseaseIndex in selectedInteractions[superVisionEdgesIndex]:
            edge_label_index[0].append(drugIndex)
            edge_label_index[1].append(diseaseIndex)
            edge_label.append(1)
        for drugIndex, diseaseIndex in selectedNonInteractions[trainNonEdgesIndex]:
            neg_edge_index[0].append(drugIndex)
            neg_edge_index[1].append(diseaseIndex)
            edge_label.append(0)
        edge_label_index = torch.tensor(edge_label_index, dtype=torch.long)
        neg_edge_index = torch.tensor(neg_edge_index, dtype=torch.long)
        edge_label = torch.tensor(edge_label, dtype=torch.float)

        data['drug', 'treats', 'disease'].edge_label_index = torch.cat([edge_label_index, neg_edge_index],dim=-1)
        data['drug', 'treats', 'disease'].edge_label = edge_label

        data = T.ToUndirected()(data)
        data = T.AddSelfLoops()(data)
        data = T.NormalizeFeatures()(data)

        model = Model(data, neurons=32, layers=LAYERS, aggregator=AGGREGATOR)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Due to lazy initialization, we need to run one model step so the number
        # of parameters can be inferred:
        with torch.no_grad():
            model.encoder(data.x_dict, data.edge_index_dict)

        criterion = torch.nn.BCEWithLogitsLoss()
        for epoch in range(1, EPOCHS):
            loss = train(data, model, optimizer, criterion)
            if epoch % 10 == 0:
                print('epoch: ', epoch, 'train loss: ', loss)
        
        #------------------Testing------------------#
        edge_label_index = [[], []]
        neg_edge_index = [[], []]
        edge_label = []
        for drugIndex, diseaseIndex in selectedInteractions[testEdgesIndex]:
            edge_label_index[0].append(drugIndex)
            edge_label_index[1].append(diseaseIndex)
            edge_label.append(1)
        for drugIndex, diseaseIndex in selectedNonInteractions[testNonEdgesIndex]:
            neg_edge_index[0].append(drugIndex)
            neg_edge_index[1].append(diseaseIndex)
            edge_label.append(0)
        edge_label_index = torch.tensor(edge_label_index, dtype=torch.long)
        neg_edge_index = torch.tensor(neg_edge_index, dtype=torch.long)
        edge_label = torch.tensor(edge_label, dtype=torch.float)
        
        data['drug', 'treats', 'disease'].edge_label_index = torch.cat([edge_label_index, neg_edge_index], dim=-1)
        data['drug', 'treats', 'disease'].edge_label = edge_label
        
        metric = test(data, model, THRESHOLD_PERCENT)
        metrics += metric

        print('metric: ', metric)
    return metrics

metrics = main()
print('results: ', metrics / FOLDS)

# def main():
#     drugDisease = np.loadtxt('./data/deepDR/drug_disease.txt', delimiter='\t')
#     drugDrug = np.loadtxt('./data/deepDR/drug_drug.txt', delimiter='\t')
#     drugProtein = np.loadtxt('./data/deepDR/drug_protein.txt', delimiter='\t')
#     drugSide = np.loadtxt('./data/deepDR/drug_sideeffect.txt', delimiter='\t')
#     totalInteractions = np.array(np.mat(np.where(drugDisease == 0)).T) #(18416, 2)

#     print('totalInteractions: ', totalInteractions.shape)
#     print('drugDisease: ', drugDisease.shape)
#     print('drugDrug: ', drugDrug.shape)
#     print('drugProtein: ', drugProtein.shape)
#     print('drugSide: ', drugSide.shape)
# main()
