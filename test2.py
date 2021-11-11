import torch
from torch._C import dtype

X = [
    [0, 0, 0],
    [1, 2, 4],
    [0, 0, 4]
]
# tensorred = torch.inverse(torch.tensor(X, dtype=float))

# print(tensorred)

import numpy as np 

ndArr = np.array(X) 
inversed = np.linalg.inv(ndArr) 
print('inversed: ', inversed)