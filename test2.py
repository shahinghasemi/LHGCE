import torch 

x = torch.tensor([1, 2, 3], dtype=float, requires_grad=True)
y = x * x * 2
z = 3 * x 
loss = y + z
print('x: ', x, 'x.grad: ', x.grad)
print('y: ', y)
loss.sum().backward()
print('y after backward: ', y, 'x.grad: ', x.grad)