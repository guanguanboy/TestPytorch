#参考：https://deeplizard.com/learn/video/kF2AlpykJGY

import torch

t1 = torch.tensor([1,1,1])
print(t1[1])
t1_unsqueeze = t1.unsqueeze(dim=0)
print(t1_unsqueeze.shape) #torch.Size([1, 3])

print(t1_unsqueeze[0]) #tensor([1, 1, 1])
#由于t1_unsqueeze是一个二维tensor，所以取第一个元素取出来是一个一维tensor
#要取出第一个元素，需要再增加一个索引
print(t1_unsqueeze[0][1])

#Now, we can also add an axis at the second index of this tensor.
t1_unsqueeze_dim1 = t1.unsqueeze(dim=1)
print(t1_unsqueeze_dim1.shape)
print(t1_unsqueeze_dim1[1]) #取出来的是一个一维tensor tensor([1])，不是一个标量tensor，tensor(1)
print(t1_unsqueeze_dim1[1][0]) #取出标量tensor：tensor(1)

