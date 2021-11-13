import torch
from torch._C import dtype 
import torch.nn as nn
import numpy as np

class GCNConv(nn.Module):
    def __init__(self, A, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.A_hat = A+torch.eye(A.size(0))
        self.D     = torch.diag(torch.sum(A,1))
        self.D     = self.D.inverse().sqrt()
        self.A_hat = torch.mm(torch.mm(self.D, self.A_hat), self.D)
        
        self.W = torch.empty(in_channels, out_channels, dtype=float)
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
        self.W = nn.Parameter(self.W)

    def forward(self, X):

        out = torch.relu(torch.mm(torch.mm(self.A_hat, X), self.W))
        dropout = nn.Dropout(p=0.4)
        out = dropout(out)
        # out = torch.nn.Dropout(0.4)
        # out = torch.nn.BatchNorm1d(out)
        return out

class GCNAE(torch.nn.Module):
    def __init__(self,A, nFeatures):
        super(GCNAE, self).__init__()
        self.Encoder = nn.Sequential(
            GCNConv(A,nFeatures, 512),
            GCNConv(A,512, 256),
            GCNConv(A,256, 128),
            GCNConv(A,128, 64)
        )
        self.Decoder = nn.Sequential(
            GCNConv(A,64, 128),
            GCNConv(A,128, 256),
            GCNConv(A,256, 512),
            GCNConv(A,512, nFeatures)
        )
        
    def forward(self,X):
        encoded = self.Encoder(X)
        reconstructed = self.Decoder(encoded)
        return reconstructed, encoded

def GCNEmbedding(A, X, nEpochs, lr):
    # we're using batch training however it's better to use mini-batch training but in Graph structure
    # I don't know how to split the data with the presence of Adjacency matrix. so for the first try I'll
    # go with the simplest approach.
    A = torch.tensor(A, dtype=float)
    X = torch.tensor(X, dtype=float)

    model = GCNAE(A, X.shape[1])
    MSELoss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    for epoch in range(nEpochs):
        batchLoss = 0
        XReconstructed, encoded = model(X)
        GCNAELoss = MSELoss(XReconstructed, X)
        batchLoss += GCNAELoss
        batchLoss.backward()
        optimizer.step()
        if epoch % 4 == 0:
            print('-> epoch: ', epoch, 'GCN batchLoss: ', batchLoss.item())

    model.train(False)
    _, encoded = model(X)
    return encoded.detach().numpy()