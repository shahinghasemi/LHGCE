
import torch

target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
print('target: ', target)
output = torch.full([10, 64], 1.5)  # A prediction (logit)
print('output: ', output)
pos_weight = torch.ones([64])  # All weights are equal to 1
print('pos_weight: ', pos_weight)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss = criterion(output, target)  # -log(sigmoid(1.5))
print('loss: ', loss)
