import torch_geometric.transforms as T
from prepareData import splitter, foldify, splitEdgesBasedOnFolds, metadata
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
parser.add_argument('--dataset', help='dataset to use', type=str, default='LAGCN')
parser.add_argument('--epochs', help='number of epochs to train the model in',type=int, default=3000)
parser.add_argument('--folds', help='number of folds',type=int, default=5)
parser.add_argument('--fold', help='specific fold to train on',type=int, default=-1)
parser.add_argument('--thr-percent', help='the threshold percentage with respect to batch size',type=float, default=3)
parser.add_argument('--lr', help='learning rate of optimizer function',type=float, default=0.001)
parser.add_argument('--l', help='number of layers for graph convolutional encoder', type=int, default=1)
parser.add_argument('--n', help='number of neurons for each GCE layer', type=int, default=32)
parser.add_argument('--same', help='whether the same number of negatives should be selected as positives(interations)', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--negative-split', help='how negatives should be involved in training and testing phase?', type=str, default='all')
parser.add_argument('--agg-lin', help='aggregator function for linear layers', type=str)
parser.add_argument('--agg-conv', help='aggregator function for conv layers', type=str)
parser.add_argument('--agg-hetero', help='aggregator function for hetero layers', type=str)
parser.add_argument('--encoder', help='encoder', type=str)

args = parser.parse_args()
print(args)

# Setting the dynamic global variables
DATASET = args.dataset
EPOCHS = args.epochs
THRESHOLD_PERCENT = args.thr_percent
LEARNING_RATE= args.lr
LAYERS = args.l
NEURONS = args.n
SAME_NEGATIVE = args.same
NEGATIVE_SPLIT = args.negative_split
FOLDS = args.folds
FOLD = args.fold
AGGREGATOR_LIN = args.agg_lin
AGGREGATOR_CONV = args.agg_conv
AGGREGATOR_HETERO = args.agg_hetero
ENCODER = args.encoder

def main():

    metrics = np.zeros(7)
    totalInteractions, totalNonInteractions, INTERACTIONS_NUMBER, NONINTERACTIONS_NUMBER = metadata(DATASET)

    selectedInteractions, selectedNonInteractions = splitter(SAME_NEGATIVE, totalInteractions, totalNonInteractions, INTERACTIONS_NUMBER, NONINTERACTIONS_NUMBER)
    interactionsIndicesFolds, nonInteractionsIndicesFolds = foldify(selectedInteractions, selectedNonInteractions)

    if FOLD != -1:
        customRange = range(FOLD, FOLD+1, 1) 
        divider = 1
    else:
        customRange = range(FOLDS)
        divider = FOLDS

    for k in customRange:
        data = dataloader(DATASET)
        messageEdgesIndex, trainSuperVisionEdgesIndex, testSuperVisionEdgesIndex = splitEdgesBasedOnFolds(interactionsIndicesFolds, k)

        testNonEdgesIndex = nonInteractionsIndicesFolds[k]
        trainNonEdgesIndex = np.setdiff1d(nonInteractionsIndicesFolds.flatten(), testNonEdgesIndex, assume_unique=True)

        # edge_index does not need to be set in testing because otherwise causes data leak
        edge_index = [[], []]
        for drugIndex, diseaseIndex in selectedInteractions[messageEdgesIndex]:
            edge_index[0].append(drugIndex)
            edge_index[1].append(diseaseIndex)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        data['drug', 'treats', 'disease'].edge_index = edge_index

        #------------------Training------------------#
        edge_label_index = [[], []]
        neg_edge_label_index = [[], []]
        edge_label = []
        for drugIndex, diseaseIndex in selectedInteractions[trainSuperVisionEdgesIndex]:
            edge_label_index[0].append(drugIndex)
            edge_label_index[1].append(diseaseIndex)
            edge_label.append(1)
        if NEGATIVE_SPLIT == 'fold':
            for drugIndex, diseaseIndex in selectedNonInteractions[trainNonEdgesIndex]:
                neg_edge_label_index[0].append(drugIndex)
                neg_edge_label_index[1].append(diseaseIndex)
                edge_label.append(0)
        elif NEGATIVE_SPLIT == 'all':
            for drugIndex, diseaseIndex in selectedNonInteractions:
                neg_edge_label_index[0].append(drugIndex)
                neg_edge_label_index[1].append(diseaseIndex)
                edge_label.append(0)
        edge_label_index = torch.tensor(edge_label_index, dtype=torch.long)
        neg_edge_label_index = torch.tensor(neg_edge_label_index, dtype=torch.long)
        edge_label = torch.tensor(edge_label, dtype=torch.float)

        data['drug', 'treats', 'disease'].edge_label_index = torch.cat([edge_label_index, neg_edge_label_index],dim=-1)
        data['drug', 'treats', 'disease'].edge_label = edge_label

        data = T.ToUndirected()(data)
        data = T.AddSelfLoops()(data)
        data = T.NormalizeFeatures()(data)
        print('hetero data: ', data)

        model = Model(data=data, neurons=NEURONS, layers=LAYERS, aggregator_lin=AGGREGATOR_LIN, aggregator_conv=AGGREGATOR_CONV, aggregator_hetero=AGGREGATOR_HETERO, encoder=ENCODER)
        print('model: ', model)
        optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

        # Due to lazy initialization, we need to run one model step so the number
        # of parameters can be inferred:
        with torch.no_grad():
            model.encoder(data.x_dict, data.edge_index_dict)

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor(NONINTERACTIONS_NUMBER / INTERACTIONS_NUMBER))
        for epoch in range(1, EPOCHS):
            loss = train(data, model, optimizer, criterion)
            if epoch % 10 == 0:
                print('epoch: ', epoch, 'train loss: ', loss)
        
        #------------------Testing------------------#
        edge_label_index = [[], []]
        neg_edge_label_index = [[], []]
        edge_label = []
        for drugIndex, diseaseIndex in selectedInteractions[testSuperVisionEdgesIndex]:
            edge_label_index[0].append(drugIndex)
            edge_label_index[1].append(diseaseIndex)
            edge_label.append(1)
        if NEGATIVE_SPLIT == 'fold':
            for drugIndex, diseaseIndex in selectedNonInteractions[testNonEdgesIndex]:
                neg_edge_label_index[0].append(drugIndex)
                neg_edge_label_index[1].append(diseaseIndex)
                edge_label.append(0)
        elif NEGATIVE_SPLIT == 'all':
            for drugIndex, diseaseIndex in selectedNonInteractions:
                neg_edge_label_index[0].append(drugIndex)
                neg_edge_label_index[1].append(diseaseIndex)
                edge_label.append(0)
        edge_label_index = torch.tensor(edge_label_index, dtype=torch.long)
        neg_edge_label_index = torch.tensor(neg_edge_label_index, dtype=torch.long)
        edge_label = torch.tensor(edge_label, dtype=torch.float)
        
        data['drug', 'treats', 'disease'].edge_label_index = torch.cat([edge_label_index, neg_edge_label_index], dim=-1)
        data['drug', 'treats', 'disease'].edge_label = edge_label
        metric = test(data, model, THRESHOLD_PERCENT)
        metrics += metric

        print("calculated metrics in fold --> " + str(k + 1)+ ": ", metric)

    return metrics / divider

metrics = main()
print("####### PARAMETERS #######", args)
print('####### FINAL RESULTS #######\n', metrics)

# extractId()