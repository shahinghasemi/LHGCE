import torch.nn as nn
import torch

loss = nn.BCELoss(reduction='none')
x = torch.tensor([[0.1], [0.1], [0.9]], dtype=float)
y = torch.tensor([[0], [0], [1]], dtype=float)
goodPred = loss(x, y)
weights = torch.tensor([[0.11], [0.11], [0.89]], dtype=float)
final_loss = torch.mean(goodPred * weights)
print(final_loss)

loss2 = nn.BCELoss()
loss2.weight = weights
goodPred = loss2(x, y)
print(final_loss)
