import torch
from torch import nn, optim
import numpy as np
from torch.nn.modules.container import Sequential
class FCNN(nn.Module):
    def __init__(self, inputShape, dropout):
        super(FCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(inputShape, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, inputShape),
            nn.ReLU(),
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


def trainFNN(dataDic, nEpochs, nBatchsize, dropout, lr):
    labels = torch.from_numpy(dataDic['y'])
    X = dataDic['X']
    model = FCNN(X.shape[1], dropout)
    # should add weighted loss
    # BCELoss = nn.BCEWithLogitsLoss()
    MSELoss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    indices = np.arange(labels.shape[0])

    for epoch in range(nEpochs):
        np.random.shuffle(indices)
        for boundary in range(0, len(indices), nBatchsize):
            batchIndex = indices[boundary:boundary + nBatchsize]
            batchLoss = 0
            XTrain = X[batchIndex].float()
            reconstructed = model(XTrain)
            dnnLoss = MSELoss(reconstructed, XTrain)
            batchLoss += dnnLoss
            optimizer.zero_grad()
            batchLoss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            print('------------------- epoch: ', epoch, ' -------------------')
            print('-> batchLoss: ', batchLoss.item())

    return model

def testFNN(model, dataDic):
    # DNN Input
    model.eval()
    MSELoss = nn.MSELoss(reduction='none')
    X = dataDic['X'].float()
    reconstructed = model(X)

    with torch.no_grad():
        loss = MSELoss(reconstructed, X)

    loss = torch.sum(loss, 1, True)
    loss =  loss / torch.sum(loss, 0)
    return loss
