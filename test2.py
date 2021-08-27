import torch

# loss = torch.nn.BCELoss(weight=torch.tensor([1, 0, 0]))
loss = torch.nn.BCELoss(reduction='none')

y_true = torch.tensor([1, 0, 0]).float()

badPred = torch.tensor([0.1, 0.9 , 0.9]).float()
goodPred = torch.tensor([0.8, 0.1 , 0.1]).float()

l_bad = loss(badPred, y_true)
l_good = loss(goodPred, y_true)

print('bad loss: ', l_bad, 'good loss: ', l_good)