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

def prepareData(featureList, embeddingMethod):
    featureMatrixDic = {}
    finalDic = {}
    for feature in featureList:
        featureMatrixDic[feature] = np.loadtxt('./data/'+ feature+ '_feature_matrix.txt')
    
    if embeddingMethod == 'AE':
        finalDic = featureMatrixDic;
    elif embeddingMethod == 'jaccard':
        for feature, matrix in featureMatrixDic.items():
            finalDic[feature] = Jaccard(matrix)
    # elif similarity == 'cosine':
    
    return finalDic
