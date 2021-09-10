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


lossWeighted = nn.BCELoss(weight=torch.tensor([0.33, 0.66, 0.66]))
lossNotWeighted = nn.BCELoss()

goodPred = torch.tensor(goodPred, dtype=torch.float64)
badPred = torch.tensor(badPred, dtype=torch.float64)

Y = torch.tensor(Y, dtype=torch.float64)
lossBad = lossWeighted(badPred, Y)
lossWeightedBad = lossNotWeighted(badPred, Y)
print('lossBad:', lossBad)
print('lossWeightedBad:', lossWeightedBad)


# m = nn.Sigmoid()
# loss = nn.BCELoss()
# input = torch.randn(3, requires_grad=True)
# print('input:', input)
# target = torch.empty(3).random_(2)
# print(target)
# output = loss(m(input), target)
# print(output)
# output.backward()


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
