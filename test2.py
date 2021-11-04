import numpy as np
np.random.seed(1)

X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
])
y = np.array([1, 0, 1, 1 ])
y_second = np.array([ 0, 0, 1])


rp = np.random.permutation(3)
print('rp: ', rp)
# X[y==1][rp[:int(len(rp)*known_labels_ratio)]]
print('[y==1]: ', [y==1])
print('X[y==1]: ', X[y==1][y_second==0])

X = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
]
print(X[1, 2, 3])