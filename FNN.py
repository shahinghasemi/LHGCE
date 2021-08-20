import torch
from torch import nn, optim
import numpy as np
from torch.nn.modules import loss


class FCNN(nn.Module):
    def __init__(self, input):
        super(FCNN, self).__init__()
        self.lin1 = nn.Linear(input, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 16)
        self.lin4 = nn.Linear(16, 4)
        self.lin5 = nn.Linear(4, 1)
        self.rel = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.rel(self.lin1(x))
        x = self.rel(self.lin2(x))
        x = self.rel(self.lin3(x))
        x = self.rel(self.lin4(x))
        x = self.sig(self.lin5(x))
        return x

def trainFNN(data, nFeatures, nEpochs, nBatchsize):
    model = FCNN(nFeatures)
    # should add weighted loss
    BCELoss = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    np.random.shuffle(data)

    for epoch in range(nEpochs):
        for iter in range(0, len(data), nBatchsize):
            batchedData = torch.tensor(data[iter:iter + nBatchsize]).float()
            X = batchedData[:, :-1]
            Y = []
            for y in batchedData[:, -1]:
                Y.append([y])
            Y = torch.tensor(Y).float()
            y_pred = model(X)
            l = BCELoss(y_pred, Y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print('epoch: ', epoch, 'loss: ', l)
    
    model.train(False)
    return model