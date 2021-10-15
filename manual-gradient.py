import torch 


X = torch.tensor([1, 2, 3, 4, 5])
Y = torch.tensor([2, 4, 6, 8, 10])
W = torch.tensor(0, dtype=float, requires_grad=True)

epochs = 100
LR = 0.001

def MSE(y, y_pred):
    return ((y - y_pred) ** 2).mean() 

def forward(x):
    return x * W;

for epoch in range(epochs):
    y_pred = forward(X)
    loss = MSE(Y, y_pred)
    loss.backward()
    W = W - (LR * W.grad)
    print('loss: ', loss, 'W: ', W, )