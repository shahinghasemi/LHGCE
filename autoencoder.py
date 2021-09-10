import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class AutoEncoder(nn.Module):
    def __init__(self, input):
        super(AutoEncoder, self).__init__()
        self.en_1 = nn.Linear(input, 512)
        self.en_2 = nn.Linear(512, 256)
        self.en_3 = nn.Linear(256, 128)
        self.en_4 = nn.Linear(128, 64)
        self.embedded = nn.Linear(64, 32)
        self.de_1 = nn.Linear(32, 64)
        self.de_2 = nn.Linear(64, 128)
        self.de_3 = nn.Linear(128, 256)
        self.de_4 = nn.Linear(256, 512)
        self.reconstructed = nn.Linear(512, input)
        self.rel = nn.ReLU()
        # dropout, batch normalization

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

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
        x = self.rel(self.reconstructed(x))
        return x;


def trainAutoEncoders(concatenatedData, nFeatures, nEpochs, nBatchsize):
    model = AutoEncoder(nFeatures)
    MSELoss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    np.random.shuffle(concatenatedData)

    for epoch in range(nEpochs):
        for boundary in range(0, len(concatenatedData), nBatchsize):
            batchedData = torch.tensor(concatenatedData[boundary:boundary + nBatchsize]).float()
            y_pred = model(batchedData)
            l = MSELoss(y_pred, batchedData)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print('epoch: ', epoch, 'loss: ', l)

    model.train(False)
    return model
