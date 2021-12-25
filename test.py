import numpy as np
import torch 


first = np.array([1, 2, 3])
second = np.array([4, 5, 6])

result = np.concatenate((first, second), axis=0)
print('result: ', result)