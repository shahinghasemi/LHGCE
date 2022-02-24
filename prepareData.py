# import matplotlib
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import torch
from torch_geometric.data import HeteroData
import pandas as pd

FOLDS = 5
# def Cosine(matrix)
def Jaccard(matrix):
    matrix = np.mat(matrix)
    numerator = matrix * matrix.T
    denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
    return np.array(numerator / denominator)

def readFromMat():
    data = sio.loadmat('./data/lagcn/SCMFDD_Dataset.mat')
    print('data.keys(): ', data.keys())

    np.savetxt('./drugDrug_feature_matrix.txt', np.array(data['drug_drug_interaction_feature_matrix']))
    np.savetxt('./structure_feature_matrix.txt', np.array(data['structure_feature_matrix']))
    np.savetxt('./target_feature_matrix.txt', np.array(data['target_feature_matrix']))
    np.savetxt('./enzyme_feature_matrix.txt', np.array(data['enzyme_feature_matrix']))
    np.savetxt('./pathway_feature_matrix.txt', np.array(data['pathway_feature_matrix']))

def plotAndSave(X, Y, labels, feature):
    print('about to draw: X[0].shape: ', X[0].shape,  'X[1].shape: ', X[1].shape)
    print('about to draw: Y[0].shape: ', Y[0].shape,  'Y[1].shape: ', Y[1].shape)

    plt.scatter(X[0], Y[0], c='blue', marker='.', linewidth=0, s=10, alpha=0.8, label=labels[0])
    plt.scatter(X[1], Y[1], c='red', marker='o', linewidth=0, s=10, alpha=0.8, label=labels[1])
    plt.grid()
    plt.legend()
    plt.show()

def splitEdgesBasedOnFolds(interactionsIndicesFolds, k):
    testEdgesIndex = interactionsIndicesFolds[k]
    # if(k+1 == FOLDS):
    #     k = 0
    # superVisionEdgesIndex = interactionsIndicesFolds[k+1]
    # usedIndices = np.concatenate((testEdgesIndex, superVisionEdgesIndex), axis=0)
    messageEdgesIndex = np.setdiff1d(interactionsIndicesFolds.flatten(), testEdgesIndex, assume_unique=True)
    superVisionEdgesIndex = messageEdgesIndex
    return messageEdgesIndex, superVisionEdgesIndex, testEdgesIndex

def splitter(dataset, sameSize, interactions, nonInteractions):
    # interactionSelectionSize = round(interactionsPercent/100 * INTERACTIONS_NUMBER)
    # nonInteractionSelectionSize = round(nonInteractionsPercent/100 * NONINTERACTIONS_NUMBER)

    if dataset == 'LAGCN':
        INTERACTIONS_NUMBER = 18416
        NONINTERACTIONS_NUMBER = 142446
    elif dataset == 'LRSSL':
        INTERACTIONS_NUMBER = 3051
        NONINTERACTIONS_NUMBER = 516552
    elif dataset == 'deepDR':
        INTERACTIONS_NUMBER = 6677
        NONINTERACTIONS_NUMBER = 1860174      
    # remove some samples to be dividable by the folds
    interactionSelectionSize = INTERACTIONS_NUMBER - (INTERACTIONS_NUMBER % FOLDS)
    nonInteractionSelectionSize = NONINTERACTIONS_NUMBER - (NONINTERACTIONS_NUMBER % FOLDS)

    # choose randomly
    interactionsIndices = np.random.choice(INTERACTIONS_NUMBER, interactionSelectionSize)
    if sameSize:
        nonInteractionsIndices = np.random.choice(NONINTERACTIONS_NUMBER, interactionSelectionSize)
    else:
        nonInteractionsIndices = np.random.choice(NONINTERACTIONS_NUMBER, nonInteractionSelectionSize)

    selectedNonInteractionsPairs = nonInteractions[nonInteractionsIndices]
    selectedInteractionsPairs = interactions[interactionsIndices]
    return selectedInteractionsPairs, selectedNonInteractionsPairs

def foldify(totalInteractions, totalNonInteractions):
    sizeOfInteractions = totalInteractions.shape[0]
    sizeOfNonInteractions = totalNonInteractions.shape[0]

    totalInteractionIndices = np.random.permutation(sizeOfInteractions)
    totalNonInteractionIndices = np.random.permutation(sizeOfNonInteractions)

    interactionsIndicesFolds = totalInteractionIndices.reshape(FOLDS, sizeOfInteractions // FOLDS)
    nonInteractionsIndicesFolds = totalNonInteractionIndices.reshape(FOLDS, sizeOfNonInteractions // FOLDS)

    return interactionsIndicesFolds, nonInteractionsIndicesFolds

def makePosEdgeIndex(dataset, name, delimiter=',', percent = 100, dataframe=False):
    if dataframe:
        matrix = pd.read_csv('./data/' + dataset + '/' + name, delimiter=delimiter, header=None).drop(columns=0, index=0).to_numpy(dtype=np.integer)
    else:
        matrix = np.loadtxt('./data/' + dataset + '/' + name, delimiter=delimiter)

    result = np.where(matrix == 1)
    edgeIndex = [[], []]
    choseIndex = np.random.choice(result[0].shape[0], int(result[0].shape[0] * percent / 100))
    for index in choseIndex:
        edgeIndex[0].append(result[0][index]) 
        edgeIndex[1].append(result[1][index])

    edgeIndex = torch.tensor(edgeIndex, dtype=torch.long)
    return edgeIndex

def createHeteroNetwork(dicNumber, featureName):
    data = HeteroData()
    # dicNumber = {
    #     'drug': DRUG_NUMBER,
    #     'disease': DISEASE_NUMBER,
    #     'pathway': PATHWAY_NUMBER,
    #     'enzyme': ENZYME_NUMBER,
    #     'structure': STRUCTURE_NUMBER,
    #     'target': TARGET_NUMBER
    # }

    data['drug'].x = torch.eye(DRUG_NUMBER, dtype=torch.float)
    data['disease'].x = torch.tensor(np.loadtxt('./data/lagcn/dis_sim.csv', delimiter=','), dtype=torch.float)

    for featureName in featureList:
        data[featureName].x = torch.eye(dicNumber[featureName], dtype=torch.float)
        data['drug', 'edge', featureName].edge_index = makePosEdgeIndex(featureName)

    return data

def prepareDrugData(featureList, embeddingMethod):
    featureMatrixDic = {}
    finalDic = {}

    for feature in featureList:
        matrix = np.loadtxt('./data/lagcn/'+ feature+ '_feature_matrix.txt')
        featureMatrixDic[feature] = matrix
    
    if embeddingMethod == 'AE' or embeddingMethod == 'matrix':
        finalDic = featureMatrixDic;

    elif embeddingMethod == 'jaccardGCN':
        for feature, matrix in featureMatrixDic.items():
            jacMatrix = Jaccard(matrix)
            # remove self loops
            embedding = GCNEmbedding(jacMatrix, matrix, 50, 0.001)
            finalDic[feature] = embedding

    elif embeddingMethod == 'jaccard':
        for feature, matrix in featureMatrixDic.items():
            finalDic[feature] = Jaccard(matrix)
            
    elif embeddingMethod == 'PCA':
        for feature, matrix in featureMatrixDic.items():
            pca = PCA(n_components=2,)
            transformed = pca.fit_transform(matrix)
            finalDic[feature] = transformed

    elif embeddingMethod == 'diseaseGCN':
        diseaseSim = np.loadtxt('./data/lagcn/dis_sim.csv', delimiter=',')
        drugDisease = np.loadtxt('./data/lagcn/drug_dis.csv', delimiter=',')
        drugDisease = np.transpose(drugDisease)
        embedding = GCNEmbedding(diseaseSim, drugDisease, 300, 0.0001)
        np.savetxt('./diseaseGCN_feature_matrix.txt', np.array(embedding))
        exit()

    else:
        exit('please provide a known embedding method')
    # elif similarity == 'cosine':
    return finalDic


