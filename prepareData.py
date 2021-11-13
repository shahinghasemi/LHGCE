import matplotlib
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from GCN import GCNEmbedding


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
    else:
        exit('please provide a known embedding method')
    # elif similarity == 'cosine':
    return finalDic


