#参考：https://deeplizard.com/learn/video/kF2AlpykJGY

import torch

t1 = torch.tensor([1,1,1])
print(t1[0])
t1_unsqueeze = t1.unsqueeze(dim=0)
print(t1_unsqueeze.shape)