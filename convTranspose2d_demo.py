import torch
import torch.nn as nn
import torch.autograd as autograd

inp = torch.ones((1, 1, 2, 2))
conv_no_pad = nn.ConvTranspose2d(1, 1, kernel_size=(3, 3), stride=2, padding=0)
conv_pad = nn.ConvTranspose2d(1, 1, kernel_size=(3, 3), stride=2, padding=1)
print(conv_no_pad(inp).shape)
print(conv_pad(inp).shape)

"""
>>> # With square kernels and equal stride
>>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
>>> # non-square kernels and unequal stride and with padding
>>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
>>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
>>> output = m(input)
>>> # exact output size can be also specified as an argument
>>> input = autograd.Variable(torch.randn(1, 16, 12, 12))
>>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
>>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
>>> h = downsample(input)
>>> h.size()
torch.Size([1, 16, 6, 6])
>>> output = upsample(h, output_size=input.size())
>>> output.size()
torch.Size([1, 16, 12, 12])
"""

#需要自己计算输出特征图大小的情况：
m = nn.ConvTranspose2d(16, 33, 3, stride=2)
m_unequal_stride = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
m_kernel_only = nn.ConvTranspose2d(16, 40, 3)

input = autograd.Variable(torch.randn(20, 16, 50, 100))
output = m(input)
print('output.size:', output.size())
output_unequal_stride = m_unequal_stride(input)
print('output_unequal_stride.size:', output_unequal_stride.size())

output_kernel_only = m_kernel_only(input)
print('output_kernel_only.size:', output_kernel_only.size())

#直接指定准确的输出特征图大小的情况
input = autograd.Variable(torch.randn(1, 16, 12, 12))
downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
h = downsample(input)
print('h.size():',h.size())

output = upsample(h, output_size=input.size())
print('output.size():', output.size())


input = torch.ones(2,2,3,3)
mm = nn.ConvTranspose2d(2, 4, kernel_size=3, stride=2,padding=1)
output = mm(input) #
print(output.shape) #torch.Size([2, 4, 5, 5])

mmm = nn.ConvTranspose2d(2, 4, kernel_size=3, stride=2,padding=1, output_padding=1)
output = mmm(input)
print(output.shape)

input2 = torch.ones(5,6, 12,12)
m_double = nn.ConvTranspose2d(in_channels=6, out_channels=16, kernel_size=3, stride=2,padding=1, output_padding=1)

output2 = m_double(input2)
print(output2.shape)