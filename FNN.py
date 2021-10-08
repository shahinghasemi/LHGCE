import torch
from torch import nn, optim
import numpy as np


class FCNN(nn.Module):
    def __init__(self, input, dropout):
        super(FCNN, self).__init__()
        self.lin1 = nn.Linear(input, 128)
        self.lin2 = nn.Linear(129, 64)
        self.lin3 = nn.Linear(64, 32)
        self.lin4 = nn.Linear(32, 16)
        self.lin5 = nn.Linear(16, 4)
        self.lin6 = nn.Linear(4, 1)
        self.rel = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.rel(self.lin1(x))
        x = self.dropout(x)
        x = self.rel(self.lin2(x))
        x = self.dropout(x)
        x = self.rel(self.lin3(x))
        x = self.dropout(x)
        x = self.rel(self.lin4(x))
        x = self.dropout(x)
        x = self.rel(self.lin5(x))
        x = self.dropout(x)
        x = self.lin6(x)
        return x

def trainFNN(data, nEpochs, nBatchsize, dropout, lr):
    # data.shape[1] contains the label too 
    model = FCNN(data.shape[1] -1 , dropout)

    # should add weighted loss
    # BCELoss = nn.BCELoss()
    # BCELoss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.114483222]))
    BCELoss = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(nEpochs):
        np.random.shuffle(data)
        for boundary in range(0, len(data), nBatchsize):
            batchedData = torch.tensor(data[boundary:boundary + nBatchsize]).float()
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
        if epoch % 5 == 0:
            print('epoch: ', epoch, 'loss: ', l)

    model.train(False)
    return model
