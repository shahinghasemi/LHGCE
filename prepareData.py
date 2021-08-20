import numpy as np
import scipy.io as sio

# def Cosine(matrix)
def Jaccard(matrix):
    matrix = np.mat(matrix)
    numerator = matrix * matrix.T
    denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
    return np.array(numerator / denominator)

def readFromMat():
    data = sio.loadmat('./data/SCMFDD_Dataset.mat')
    print(data.keys())
    np.savetxt('./structure_feature_matrix.txt', np.array(data['structure_feature_matrix']))
    np.savetxt('./target_feature_matrix.txt', np.array(data['target_feature_matrix']))
    np.savetxt('./enzyme_feature_matrix.txt', np.array(data['enzyme_feature_matrix']))
    np.savetxt('./pathway_feature_matrix.txt', np.array(data['pathway_feature_matrix']))


def concatenation(drugSimilarity, diseaseSimilarity, indices):
    concatenatedData = []
    # loop through all the interactions indices(first is for drugs and second for disease)
    for pair in indices:
        # choose involved drug and disease vectors and concatenate them
        concatenatedData.append(np.hstack((drugSimilarity[pair[0]], diseaseSimilarity[pair[1]])))
    return concatenatedData

def prepareData():
    interactionData = []
    nonInteractionData = []
    # read drug similarity features
    structure = np.loadtxt('./data/structure_feature_matrix.txt')
    target = np.loadtxt('./data/target_feature_matrix.txt')
    enzyme = np.loadtxt('./data/enzyme_feature_matrix.txt')
    pathway = np.loadtxt('./data/pathway_feature_matrix.txt')

    # read disease similarity feature
    diseaseSim = np.loadtxt('./data/dis_sim.csv', delimiter=',')
    # read the interactions matrix
    drugDisease = np.loadtxt('./data/drug_dis.csv', delimiter=',')

    # compute Jaccard similarity for each drug feature
    structureSim = Jaccard(structure)
    targetSim = Jaccard(target)
    enzymeSim = Jaccard(enzyme)
    pathwaySim = Jaccard(pathway)

    # the pair indices of the interactions [[drugIndex, diseaseIndex], [] ... []]
    # interactionIndices = np.array(np.mat(np.where(drugDisease == 1)).T)
    # nonInteractionIndices = np.array(np.mat(np.where(drugDisease == 0)).T)

    # concatenate the drugs data and disease data for involved pair drug-disease in interactions
    # interactionData.append(concatenation(structureSim, diseaseSim, interactionIndices))
    # interactionData.append(concatenation(targetSim, diseaseSim, interactionIndices))
    # interactionData.append(concatenation(enzymeSim, diseaseSim, interactionIndices))
    # interactionData.append(concatenation(pathwaySim, diseaseSim, interactionIndices))

    # nonInteractionData.append(concatenation(structureSim, diseaseSim, nonInteractionIndices))
    # nonInteractionData.append(concatenation(targetSim, diseaseSim, nonInteractionIndices))
    # nonInteractionData.append(concatenation(enzymeSim, diseaseSim, nonInteractionIndices))
    # nonInteractionData.append(concatenation(pathwaySim, diseaseSim, nonInteractionIndices))
    
    # return an array of shape (nOfDrugsFeatures, nOfInteractions, 269 + 598) -> (4, 18416, 867) for interactionData
    # return an array of shape (nOfDrugsFeatures, nOfNonInteractions, 269 + 598) -> (4, 142446, 867) for interactionData
    return {
        'structure': structureSim,
        'target': targetSim,
        'enzyme': enzymeSim,
        'pathway': pathwaySim
    }
