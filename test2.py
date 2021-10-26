import numpy as np

def Jaccard(matrix):
    matrix = np.mat(matrix)
    numerator = matrix * matrix.T
    denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
    return np.array(numerator / denominator)
    
def matrix_normalize(similarity_matrix = None): 
    similarity_matrix[np.isnan[similarity_matrix]] = 0
    row,col = similarity_matrix.shape
    for i in np.arange(1,row+1).reshape(-1):
        similarity_matrix[i,i] = 0
    
    if row == col:
        similarity_matrix[np.isnan[similarity_matrix]] = 0
        for i in np.arange(1,similarity_matrix.shape[1-1]+1).reshape(-1):
            similarity_matrix[i,i] = 0
        for round in np.arange(1,200+1).reshape(-1):
            D = diag(np.sum(similarity_matrix, 2-1))
            D1 = pinv(np.sqrt(D))
            similarity_matrix = D1 * similarity_matrix * D1
    else:
        for j in np.arange(1,similarity_matrix.shape[1-1]+1).reshape(-1):
            if sum(similarity_matrix(j,:)) != 0:
                similarity_matrix[j,:] = similarity_matrix(j,:) / sum(similarity_matrix(j,:))
            else:
                similarity_matrix[j,:] = np.zeros((1,similarity_matrix.shape[2-1]))
    
    return similarity_matrix

x  = np.array([
    [0, 0, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 0]
])

jaccard = Jaccard(x)