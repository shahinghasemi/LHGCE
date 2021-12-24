import matplotlib.pyplot as plt 
import numpy as np
import torch 

z = torch.tensor([
    [2, 1], 
    [6, 3],
    [4, 5],
    [7, 8]
], dtype=float)


value = (z[[0, 1, 0, 2]] * z[[1, 0, 0, 3]])
print('value: ', value)