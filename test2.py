import torch
loss = torch.tensor([[1], [2], [3]])
summed = loss / torch.sum(loss, 0)
print('summed: ', summed)