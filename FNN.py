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
        elif mode == 'decoder':
            if featureName == 'target':
                return self.target_decoder(x)
            elif featureName == 'structure':
                return self.structure_decoder(x)
            elif featureName == 'enzyme':
                return self.enzyme_decoder(x)
            elif featureName == 'pathway':
                return self.pathway_decoder(x)

        elif mode == 'DNN':
            return self.DNN(x)


def trainFNN(dataDic, emb, nEpochs, nBatchsize, dropout, lr, featuresList, aggregationMode):
    inputs = {}
    for key, value in dataDic.items():
        if key != 'labels' and key != 'diseases':
            inputs[key] = value.shape[1]

    model = FCNN(inputs, emb, dropout, aggregationMode)
    # should add weighted loss
    BCELoss = nn.BCEWithLogitsLoss()
    MSELoss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    indices = np.arange(dataDic['labels'].shape[0])

    for epoch in range(nEpochs):
        np.random.shuffle(indices)
        for boundary in range(0, len(indices), nBatchsize):
            batchIndex = indices[boundary:boundary + nBatchsize]
            encodedDic = {}
            decodedDic = {}
            featureLossesDic = {}
            batchLoss = 0
            for feature in featuresList:
                X = torch.tensor(dataDic[feature][batchIndex]).float()
                encoded = model(X, feature, 'encoder')
                decoded = model(encoded, feature, 'decoder')
                encodedDic[feature] = encoded
                decodedDic[feature] = decoded
            
            # DNN Input
            DnnInput = torch.tensor([], dtype=float)
            for index, key in enumerate(encodedDic):
                if index == 0:
                    DnnInput = encodedDic[key]
                else:
                    if aggregationMode == 'concatenate':
                        DnnInput = torch.cat((DnnInput, encodedDic[key]), 1)

            # AE Losses
            for index, key in enumerate(decodedDic):
                X = torch.tensor(dataDic[key][batchIndex]).float()
                featureLossesDic[key] = MSELoss(decodedDic[key], X)
                batchLoss += featureLossesDic[key]

            Y = torch.tensor(dataDic['labels'][batchIndex]).float()
            y_pred = model(DnnInput, None, 'DNN')

            dnnLoss = BCELoss(y_pred, Y) 
            batchLoss += dnnLoss
            optimizer.zero_grad()
            batchLoss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            print('------------------- epoch: ', epoch, ' -------------------' )
            print('-> batchLoss: ', batchLoss.item())
            print('-> dnnLoss: ', dnnLoss.item())
            # AE Losses
            for index, key in enumerate(featureLossesDic):
                print(key, ' loss: ', featureLossesDic[key].item())


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
