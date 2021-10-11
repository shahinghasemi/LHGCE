import torch
from torch import nn, optim
import numpy as np


class FCNN(nn.Module):
    def __init__(self, input, emb, dropout):
        super(FCNN, self).__init__()
        self.en_1 = nn.Linear(input, 512)
        self.en_2 = nn.Linear(512, 256)
        self.en_3 = nn.Linear(256, 128)
        self.en_4 = nn.Linear(128, 64)
        self.embedded = nn.Linear(64, emb)
        self.de_1 = nn.Linear(32, 64)
        self.de_2 = nn.Linear(64, 128)
        self.de_3 = nn.Linear(128, 256)
        self.de_4 = nn.Linear(256, 512)
        self.reconstruct = nn.Linear(512, input)

        self.lin1 = nn.Linear(emb, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 32)
        self.lin4 = nn.Linear(32, 16)
        self.lin5 = nn.Linear(16, 4)
        self.lin6 = nn.Linear(4, 1)

        self.rel = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def encode(self, x, convertToNumpy=False):
        x = self.rel(self.en_1(x))
        x = self.rel(self.en_2(x))
        x = self.rel(self.en_3(x))
        x = self.rel(self.en_4(x))
        x = self.rel(self.embedded(x))
        if convertToNumpy:
            # return torch.tensor(x.detach(), dtype=torch.float32).numpy()
            return x.detach().numpy()
        return x
    
    def decode(self, x):
        x = self.rel(self.de_1(x))
        x = self.rel(self.de_2(x))
        x = self.rel(self.de_3(x))
        x = self.rel(self.de_4(x))
        x = self.rel(self.reconstruct(x))
        return x;
        
    def forward(self, x):
        embedded = self.encode(x)
        reconstructed = self.decode(embedded)
        x = self.DNN(embedded)
        return x, reconstructed

    def DNN(self, x):
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

def trainFNN(data, emb, nEpochs, nBatchsize, dropout, lr):
    # data.shape[1] contains the label too 
    model = FCNN(data.shape[1] -1, emb, dropout)

    # should add weighted loss
    # BCELoss = nn.BCELoss()
    # BCELoss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.114483222]))
    BCELoss = nn.BCEWithLogitsLoss()
    MSELoss = nn.MSELoss()

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
            y_pred, reconstructed = model(X)
            # don't know whether use Y or X for MSELoss
            l = BCELoss(y_pred, Y) + MSELoss(reconstructed, X)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print('epoch: ', epoch, 'loss: ', l)

    model.train(False)
    return model
