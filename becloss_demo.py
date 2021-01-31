
import torch
import torch.nn as nn

input = torch.FloatTensor([[ -0.4089,  -1.2471,  0.5907],
        [ -0.4897, -0.8267,  -0.7349],
        [ 0.5241,  -0.1246, -0.4751]])
print(input)
target = torch.FloatTensor([[0, 1, 1],
                            [0, 0, 1],
                            [1, 0, 1]])

m = nn.Sigmoid()
sigmoid_res = m(input)
loss = nn.BCELoss()
# 公式为-1/n * （累加（y*lnx + （1-y）* ln（1-x）））
res = loss(sigmoid_res, target)
print(res)

#BCEWithLogitsLoss就是把Sigmoid-BCELoss合成一步。
loss1 = nn.BCEWithLogitsLoss()
res1 = loss1(input, target)
print(res1) #tensor(0.7193)