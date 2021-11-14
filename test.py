import matplotlib.pyplot as plt 
import numpy as np
import torch 

tensor = torch.tensor([
    [2, 1], 
    [6, 3]
], dtype=float)

print(tensor.inverse())
print(torch.linalg.inv(tensor))
print(torch.linalg.pinv(tensor))