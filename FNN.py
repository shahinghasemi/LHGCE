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
    
        self.DNN = Sequential(
            nn.Linear(self.inputDim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4, 1),
        )
        
    def forward(self, x, featureName, mode):
        if mode == 'DNN':
            return self.DNN(x)


def trainFNN(dataDic, nEpochs, nBatchsize, dropout, lr, featuresList, aggregationMode):
    inputsShape = {}
    labels = torch.from_numpy(dataDic['labels'])
    del dataDic['labels']

    for key, value in dataDic.items():
        inputsShape[key] = value.shape[1]

    model = FCNN(inputsShape, dropout, aggregationMode)
    # should add weighted loss
    BCELoss = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    indices = np.arange(labels.shape[0])

    for epoch in range(nEpochs):
        np.random.shuffle(indices)
        for boundary in range(0, len(indices), nBatchsize):
            batchIndex = indices[boundary:boundary + nBatchsize]
            batchLoss = 0
            # DNN Input
            DnnInput = torch.tensor([], dtype=float)
            for index, featureKey in enumerate(dataDic):
                tensorred = torch.from_numpy(dataDic[featureKey][batchIndex]).float()
                if index == 0:
                    DnnInput = tensorred
                else:
                    if aggregationMode == 'concatenate':
                        DnnInput = torch.cat((DnnInput, tensorred), 1)

            Y = torch.tensor(labels[batchIndex]).float()
            y_pred = model(DnnInput, None, 'DNN')

            dnnLoss = BCELoss(y_pred, Y) 
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
    DnnInput = torch.tensor([], dtype=float)
    for index, featureKey in enumerate(dataDic):
        tensorred = torch.from_numpy(dataDic[featureKey]).float()
        if index == 0:
            DnnInput = tensorred
        else:
            if aggregationMode == 'concatenate':
                DnnInput = torch.cat((DnnInput, tensorred), 1)    
        
    y_pred = model(DnnInput, None, 'DNN')
    sig = torch.nn.Sigmoid()
    return sig(y_pred)
