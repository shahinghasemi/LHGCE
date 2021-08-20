import torch
from torch import tensor
import torch.nn as nn
import numpy as np 

def binaryCrossEntropyLoss(pred, Y, weights):
    firstPart = Y * np.log(pred)
    print('firstPart: ', firstPart)
    secondPart = (1 - Y) * np.log(1 - pred)
    print('secondPart: ', secondPart)
    summation = weights[0] * firstPart + weights[1] * secondPart
    print('summation: ', summation)
    return - np.mean(summation)

Y = np.array([1, 0, 0])
goodPred = np.array([0.95, 0.025, 0.025])
badPred = np.array([0.20, 0.68, 0.12])


loss = nn.BCELoss(weight=torch.tensor([1.5, 1]))

goodPred = torch.tensor(goodPred)
Y = torch.tensor(Y)
print(Y)
print(goodPred)
loss(goodPred, Y)

# binaryCrossEntropyLoss(goodPred, Y, [1, 1])
# print('------------------')
# binaryCrossEntropyLoss(badPred, Y, [1, 1])
# print('------------------')
# print('::::::weighted::::::')
# binaryCrossEntropyLoss(goodPred, Y, [1.5, 1])
# print('------------------')
# binaryCrossEntropyLoss(badPred, Y, [1.5, 1])
# print('------------------')
# print('::::::weighted::::::')
