# import matplotlib
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
import torch
from GCN import GCNEmbedding
from torch_geometric.data import HeteroData

DRUG_NUMBER = 269
DISEASE_NUMBER = 598
ENZYME_NUMBER = 108
STRUCTURE_NUMBER = 881
PATHWAY_NUMBER = 258
TARGET_NUMBER = 529
INTERACTIONS_NUMBER = 18416
NONINTERACTIONS_NUMBER = 142446
FOLDS = 5


# def Cosine(matrix)
def Jaccard(matrix):
    matrix = np.mat(matrix)
    numerator = matrix * matrix.T
    denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
    return np.array(numerator / denominator)

def readFromMat():
    data = sio.loadmat('./data/SCMFDD_Dataset.mat')
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
    if(k+1 == FOLDS):
        k = 0
    # superVisionEdgesIndex = interactionsIndicesFolds[k+1]
    # usedIndices = np.concatenate((testEdgesIndex, superVisionEdgesIndex), axis=0)
    messageEdgesIndex = np.setdiff1d(interactionsIndicesFolds.flatten(), testEdgesIndex, assume_unique=True)
    superVisionEdgesIndex = messageEdgesIndex
    return messageEdgesIndex, superVisionEdgesIndex, testEdgesIndex

def splitter(interactionsPercent, nonInteractionsPercent, interactions, nonInteractions, folds=5):
    interactionSelectionSize = round(interactionsPercent/100 * INTERACTIONS_NUMBER)
    nonInteractionSelectionSize = round(nonInteractionsPercent/100 * NONINTERACTIONS_NUMBER)

    # remove some samples to be dividable by the folds
    interactionSelectionSize = interactionSelectionSize - (interactionSelectionSize % folds)
    nonInteractionSelectionSize = nonInteractionSelectionSize - (nonInteractionSelectionSize % folds)

    # choose randomly
    interactionsIndices = np.random.choice(INTERACTIONS_NUMBER, interactionSelectionSize)
    nonInteractionsIndices = np.random.choice(NONINTERACTIONS_NUMBER, nonInteractionSelectionSize)

    selectedNonInteractions = nonInteractions[nonInteractionsIndices]
    selectedInteractions = interactions[interactionsIndices]
    return selectedInteractions, selectedNonInteractions

def foldify(totalInteractions, totalNonInteractions):
    sizeOfInteractions = totalInteractions.shape[0]
    sizeOfNonInteractions = totalNonInteractions.shape[0]

    totalInteractionIndices = np.random.permutation(sizeOfInteractions)
    totalNonInteractionIndices = np.random.permutation(sizeOfNonInteractions)

    interactionsIndicesFolds = totalInteractionIndices.reshape(FOLDS, sizeOfInteractions // FOLDS)
    nonInteractionsIndicesFolds = totalNonInteractionIndices.reshape(FOLDS, sizeOfNonInteractions // FOLDS)

    return interactionsIndicesFolds, nonInteractionsIndicesFolds

def makeNegEdgeIndex(name, isCSV=False):
    if isCSV:
        matrix = np.loadtxt('./data/drug_' + name + '.csv', delimiter=',')
    else:
        matrix = np.loadtxt('./data/drug_' + name + '.txt')
    result = np.where(matrix == 0)
    edgeIndex = [[], []]
    for index in result[0]:
        edgeIndex[0].append(index)
    for index in result[1]:
        edgeIndex[1].append(index)

    edgeIndex = torch.tensor(edgeIndex, dtype=torch.long)
    return edgeIndex

def makePosEdgeIndex(name, isCSV=False):
    if isCSV:
        matrix = np.loadtxt('./data/drug_' + name + '.csv', delimiter=',')
    else:
        matrix = np.loadtxt('./data/drug_' + name + '.txt')
    result = np.where(matrix == 1)
    edgeIndex = [[], []]
    for index in result[0]:
        edgeIndex[0].append(index)
    for index in result[1]:
        edgeIndex[1].append(index)

    edgeIndex = torch.tensor(edgeIndex, dtype=torch.long)
    return edgeIndex

def createHeteroNetwork(keys):
    data = HeteroData()

    data['drug'].x = torch.eye(DRUG_NUMBER, dtype=torch.float)
    data['disease'].x = torch.tensor(np.loadtxt('./data/dis_sim.csv', delimiter=','), dtype=torch.float)
    data['pathway'].x = torch.eye(PATHWAY_NUMBER, dtype=torch.float)
    data['enzyme'].x = torch.eye(ENZYME_NUMBER, dtype=torch.float)
    data['structure'].x = torch.eye(STRUCTURE_NUMBER, dtype=torch.float)
    data['target'].x = torch.eye(TARGET_NUMBER, dtype=torch.float)

    for key in keys:
        data['drug', 'edge', key].edge_index = makePosEdgeIndex(key)

    return data

def prepareDrugData(featureList, embeddingMethod):
    featureMatrixDic = {}
    finalDic = {}

    for feature in featureList:
        matrix = np.loadtxt('./data/'+ feature+ '_feature_matrix.txt')
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
        diseaseSim = np.loadtxt('./data/dis_sim.csv', delimiter=',')
        drugDisease = np.loadtxt('./data/drug_dis.csv', delimiter=',')
        drugDisease = np.transpose(drugDisease)
        embedding = GCNEmbedding(diseaseSim, drugDisease, 300, 0.0001)
        np.savetxt('./diseaseGCN_feature_matrix.txt', np.array(embedding))
        exit()

    else:
        exit('please provide a known embedding method')
    # elif similarity == 'cosine':
    return finalDic


