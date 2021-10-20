import torch
from torch import nn, optim
import numpy as np
from torch.nn.modules.container import Sequential

class FCNN(nn.Module):
    def __init__(self, inputs, emb, dropout, aggregationMode):
        super(FCNN, self).__init__()
        self.numFeatures = len(inputs.keys())
        self.aggregationMode = aggregationMode

        for name, inputDim in inputs.items():
            scopeNameEncoder = name + '_encoder'
            scopeNameDecoder = name + '_decoder'
            encoder = nn.Sequential(
                nn.Linear(inputDim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, emb),
                nn.ReLU(),
            )
            decoder = nn.Sequential(
                nn.Linear(emb, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, inputDim),
                nn.ReLU(),
            )
            setattr(self, scopeNameEncoder, encoder)
            setattr(self, scopeNameDecoder, decoder)

        self.DNN = Sequential(
            nn.Linear( self.numFeatures * emb if self.aggregationMode == 'concatenate' else emb, 16),
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
        if mode == 'encoder':
            if featureName == 'target':
                return self.target_encoder(x)
            elif featureName == 'structure':
                return self.structure_encoder(x)
            elif featureName == 'enzyme':
                return self.enzyme_encoder(x)
            elif featureName == 'pathway':
                return self.pathway_encoder(x)

        elif mode == 'DNN':
            return self.DNN(x)


def trainFNN(dataDic, emb, nEpochs, nBatchsize, dropout, lr, featuresList, aggregationMode):
    inputs = {}
    for key, value in dataDic.items():
        if key != 'labels' and key != 'diseases':
            inputs[key] = value.shape[1]

    model = FCNN(inputs, emb, dropout, aggregationMode)
    posWeight = torch.tensor(142000 / 12000) 
    # should add weighted loss
    BCELoss = nn.BCEWithLogitsLoss(pos_weight=posWeight)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    indices = np.arange(dataDic['labels'].shape[0])

    for epoch in range(nEpochs):
        np.random.shuffle(indices)
        for boundary in range(0, len(indices), nBatchsize):
            batchIndex = indices[boundary:boundary + nBatchsize]
            encodedDic = {}
            for feature in featuresList:
                X = torch.tensor(dataDic[feature][batchIndex]).float()
                encoded = model(X, feature, 'encoder')
                encodedDic[feature] = encoded
            
            DnnInput = torch.tensor([], dtype=float)
            for index, key in enumerate(encodedDic):
                if index == 0:
                    DnnInput = encodedDic[key]
                else:
                    if aggregationMode == 'concatenate':
                        DnnInput = torch.cat((DnnInput, encodedDic[key]), 1)
        
            Y = torch.tensor(dataDic['labels'][batchIndex]).float()
            y_pred = model(DnnInput, None, 'DNN')

            loss = BCELoss(y_pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            print('epoch: ', epoch, 'loss: ', loss)

    model.eval()
    return model

def testFNN(model, dataDic, featuresList, aggregationMode):
    encodedDic = {}
    for feature in featuresList:
        X = torch.tensor(dataDic[feature]).float()
        encoded = model(X, feature, 'encoder')
        encodedDic[feature] = encoded
    
    DnnInput = torch.tensor([], dtype=float)
    for index, key in enumerate(encodedDic):
        if index == 0:
            DnnInput = encodedDic[key]
        else:
            if aggregationMode == 'concatenate':
                DnnInput = torch.cat((DnnInput, encodedDic[key]), 1)
        
    y_pred = model(DnnInput, None, 'DNN')
    sig = torch.nn.Sigmoid()
    return sig(y_pred)
