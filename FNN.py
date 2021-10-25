import torch
from torch import nn, optim
import numpy as np
from torch.nn.modules.container import Sequential
class FCNN(nn.Module):
    def __init__(self, inputsShape, dropout, aggregationMode):
        super(FCNN, self).__init__()
        self.inputs = inputsShape
        self.inputDim = 0
        self.aggregationMode = aggregationMode

        for featureKey, shape in self.inputs.items():
            if self.aggregationMode == 'concatenate':
                self.inputDim += shape

            self.encoder = nn.Sequential(
                nn.Linear(self.inputDim, 128),
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
                nn.Linear(128, self.inputDim),
                nn.ReLU(),
            )
        
    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


def trainFNN(dataDic, nEpochs, nBatchsize, dropout, lr, featuresList, aggregationMode):
    inputsShape = {}
    labels = torch.from_numpy(dataDic['labels'])
    del dataDic['labels']

    for key, value in dataDic.items():
        inputsShape[key] = value.shape[1]

    model = FCNN(inputsShape, dropout, aggregationMode)
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
            # DNN Input
            X = torch.tensor([], dtype=float)
            for index, featureKey in enumerate(dataDic):
                tensorred = torch.from_numpy(dataDic[featureKey][batchIndex]).float()
                if index == 0:
                    X = tensorred
                else:
                    if aggregationMode == 'concatenate':
                        X = torch.cat((X, tensorred), 1)

            reconstructed = model(X)

            dnnLoss = MSELoss(reconstructed, X)
            batchLoss += dnnLoss
            optimizer.zero_grad()
            batchLoss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            print('------------------- epoch: ', epoch, ' -------------------' )
            print('-> batchLoss: ', batchLoss.item())

    model.eval()
    return model

def testFNN(model, dataDic, featuresList, aggregationMode):
    # DNN Input
    MSELoss = nn.MSELoss(reduction='none')
    X = torch.tensor([], dtype=float)
    print('here')
    for index, featureKey in enumerate(dataDic):
        tensorred = torch.from_numpy(dataDic[featureKey]).float()
        if index == 0:
            X = tensorred
        else:
            if aggregationMode == 'concatenate':
                X = torch.cat((X, tensorred), 1)    
    reconstructed = model(X)
    print('here2')

    with torch.no_grad():
        loss = MSELoss(reconstructed, X)

    print('here3')
    loss = torch.sum(loss, 1, True)
    loss =  loss / torch.sum(loss, 0)
    return loss
